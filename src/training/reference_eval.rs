//! Read-only CUDA evaluation of a native checkpoint on reserved reference text.
use super::{TokenDataLoader, Trainer};
use crate::{AppResult, reference_validation as reference};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use std::{fs, path::Path, time::Instant};
mod answers;
mod causality;
mod input;
mod repeatability;
mod replay_provenance;
pub(in crate::training) use answers::run as run_answers;
pub(in crate::training) use replay_provenance::expectation as replay_expectation;
mod seen;
#[cfg(test)]
mod trained_causality_tests;

pub(in crate::training) fn run(
    checkpoint: &Path,
    documents: &Path,
    output: Option<&Path>,
    replay_record: Option<&Path>,
    requested_feedback_passes: Option<usize>,
) -> AppResult {
    // Parse and identify every input before creating a CUDA context.
    if !checkpoint.is_file() {
        return Err(format!("checkpoint does not exist: {}", checkpoint.display()).into());
    }
    let windows = input::load(documents)?;
    let checkpoint_sha = reference::sha256(checkpoint)?;
    let output = output.map(Path::to_path_buf).unwrap_or_else(|| {
        let filename = requested_feedback_passes.map_or_else(
            || "reference_validation.json".to_owned(),
            |passes| format!("reference_validation_p{passes}.json"),
        );
        checkpoint
            .parent()
            .unwrap_or(Path::new("."))
            .join(filename)
    });
    let checkpoint_audit = crate::checkpoint::audit_model(checkpoint)?;
    let audit_path = output.with_extension("quantization.json");
    if let Some(parent) = audit_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &audit_path,
        serde_json::to_string_pretty(&checkpoint_audit)? + "\n",
    )?;
    println!("checkpoint_quantization_audit={}", audit_path.display());
    if checkpoint_audit["nonfinite_values"].as_u64().unwrap_or(1) != 0 {
        return Err("nonfinite represented checkpoint values; quantization audit saved".into());
    }
    let started = Instant::now();
    let mut trainer = Trainer::from_checkpoint(0x47505432, checkpoint)?;
    if requested_feedback_passes.is_some() && trainer.latent_feedback.is_none() {
        return Err("--feedback-evaluation-passes requires a latent-feedback checkpoint".into());
    }
    let default_passes = trainer.evaluation_passes();
    let passes = requested_feedback_passes.unwrap_or(default_passes);
    println!("reference_checkpoint={}", checkpoint.display());
    println!("reference_evaluation_passes={passes}");

    // The saved checkpoint contains the already-materialized evaluation weights.
    // Do not interpolate or remap them through newly initialized optimizer state.
    let validation = TokenDataLoader::fineweb_validation_tokens()?;
    let native_ce = trainer.eval_loss_windows(&validation, super::data::VALIDATION_WINDOWS)?;
    let native_batch = trainer.batch_from_windows(&validation, super::data::VALIDATION_WINDOWS)?;
    let token_losses = trainer.eval_token_losses(&native_batch)?;
    if token_losses.iter().any(|loss| !loss.is_finite()) {
        return Err("nonfinite native checkpoint replay".into());
    }
    let tokenwise_ce =
        token_losses.iter().map(|&v| f64::from(v)).sum::<f64>() / token_losses.len() as f64;
    if (f64::from(native_ce) - tokenwise_ce).abs() > 1e-4 {
        return Err(
            "native checkpoint tokenwise replay disagrees with the existing evaluator".into(),
        );
    }
    let (expected_ce, verified_replay_record) = replay_expectation(checkpoint, replay_record)?;
    if expected_ce.is_some_and(|value| (f64::from(native_ce) - value).abs() > 1e-4) {
        return Err(format!(
            "checkpoint replay CE {native_ce} differs from recorded final CE {expected_ce:?}"
        )
        .into());
    }
    println!("reference_native_replay_ce={native_ce:.9} tokenwise_ce={tokenwise_ce:.9}");
    // Producer CE is defined at the default depth. The repeatability and
    // causality checks below follow the selected depth of the full score.
    let selected_losses = if passes == default_passes {
        token_losses
    } else {
        trainer.eval_token_losses_for_passes(&native_batch, passes)?
    };
    let selected_native_ce =
        selected_losses.iter().map(|&v| f64::from(v)).sum::<f64>() / selected_losses.len() as f64;
    let repeatability = repeatability::verify(&mut trainer, &validation, &selected_losses, passes)?;
    let repeatability_path = output.with_extension("repeatability.json");
    fs::write(
        &repeatability_path,
        serde_json::to_string_pretty(&repeatability)? + "\n",
    )?;
    println!("reference_repeatability={repeatability}");
    if !repeatability["passed"].as_bool().unwrap_or(false) {
        return Err(format!(
            "native checkpoint repeatability failed; details saved to {}",
            repeatability_path.display()
        )
        .into());
    }
    let causality = causality::verify(&mut trainer, &windows, passes)?;
    let seen_data_replay = seen::score(&mut trainer, checkpoint, passes)?;

    let mut nats = 0_f64;
    let mut targets = 0_usize;
    let mut batches = 0;
    let scoring_started = Instant::now();
    for group in windows.chunks(GPT2_BATCH_SIZE) {
        let mut flat = group
            .iter()
            .flat_map(|window| window.ids.iter().copied())
            .collect::<Vec<_>>();
        flat.resize(GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1), 0);
        let batch = trainer.batch_from_windows(&flat, GPT2_BATCH_SIZE)?;
        let losses = trainer.eval_token_losses_for_passes(&batch, passes)?;
        for (sequence, window) in group.iter().enumerate() {
            for &loss in
                &losses[sequence * GPT2_SEQ_LEN + window.lo..sequence * GPT2_SEQ_LEN + window.hi]
            {
                if !loss.is_finite() {
                    return Err(format!(
                        "nonfinite reference target after {targets} scored targets"
                    )
                    .into());
                }
                nats += f64::from(loss);
                targets += 1;
            }
        }
        batches += 1;
        if batches % 100 == 0 {
            println!(
                "reference_validation_progress batches={batches} targets={targets}/{} elapsed_s={:.3}",
                reference::TARGETS,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let scoring_elapsed_s = scoring_started.elapsed().as_secs_f64();
    if targets != reference::TARGETS || trainer.buffers.optimizer_state.step != 0 {
        return Err("reference evaluation changed optimizer state or target accounting".into());
    }
    let bpb = nats / reference::RAW_BYTES as f64 / std::f64::consts::LN_2;
    let reference_bpb = 3.28 * reference::SOURCE_TARGETS as f64
        / reference::RAW_BYTES as f64
        / std::f64::consts::LN_2;
    let result = serde_json::json!({
        "status": "completed",
        "all_scored_losses_finite": true,
        "optimizer_updates": 0,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "document_file": documents,
        "document_file_sha256": reference::DATA_SHA,
        "raw_text_sha256": reference::RAW_SHA,
        "tokenizer_sha256": reference::TOKENIZER_SHA,
        "source_sha256": reference::SOURCE_SHA,
        "vocab_sha256": reference::VOCAB_SHA,
        "documents": reference::DOCUMENTS,
        "official_source_target_tokens": reference::SOURCE_TARGETS,
        "raw_target_bytes": reference::RAW_BYTES,
        "scored_target_tokens": targets,
        "loss_sum_nats": nats,
        "mean_token_ce": nats / targets as f64,
        "bits_per_byte": bpb,
        "context_tokens": reference::CONTEXT,
        "sliding_stride_tokens": reference::STRIDE,
        "batch_size": GPT2_BATCH_SIZE,
        "windows": windows.len(),
        "batches": batches,
        "evaluation_passes": passes,
        "default_evaluation_passes": default_passes,
        "selected_depth_native_ce": selected_native_ce,
        "scoring_elapsed_s": scoring_elapsed_s,
        "native_checkpoint_replay_ce": native_ce,
        "tokenwise_native_replay_ce": tokenwise_ce,
        "recorded_final_native_ce": expected_native_ce(checkpoint)?,
        "native_replay_expected_ce":expected_ce,
        "verified_replay_provenance":verified_replay_record,
        "causality": causality,
        "checkpoint_quantization_audit": audit_path,
        "repeatability": repeatability,
        "seen_data_replay": seen_data_replay,
        "elapsed_s": started.elapsed().as_secs_f64(),
        "reference_target_bpb_calculated": reference_bpb,
        "bpb_relative_to_calculated_target": bpb / reference_bpb - 1.0,
        "comparison_limits": "Same raw benchmark text; document resets, context/window conventions, architecture, precision and training budgets differ. Reference BPB is calculated from CE 3.28, not a newly measured reference-model result.",
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_string_pretty(&result)? + "\n")?;
    println!("reference_eval_result={}", output.display());
    println!("{result}");
    Ok(())
}

pub(in crate::training) fn expected_native_ce(checkpoint: &Path) -> AppResult<Option<f64>> {
    let Some(parent) = checkpoint.parent() else {
        return Ok(None);
    };
    let path = parent.join("metrics/valid/epoch-1/Validation_loss.log");
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)?;
    let last = content
        .lines()
        .last()
        .ok_or("empty native validation loss log")?;
    Ok(Some(
        last.split(',')
            .next()
            .ok_or("invalid native validation log")?
            .parse()?,
    ))
}
