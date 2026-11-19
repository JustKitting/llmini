//! Native fixed-checkpoint evaluation-depth diagnostic, with no updates.
use super::{TokenDataLoader, Trainer};
use crate::{AppResult, reference_validation};
use serde_json::json;
use std::{fs, path::Path};

pub(super) fn run(
    checkpoint: &Path,
    output: Option<&Path>,
    replay_record: Option<&Path>,
) -> AppResult {
    if !checkpoint.is_file() {
        return Err(format!("checkpoint does not exist: {}", checkpoint.display()).into());
    }
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        return Err("feedback depth inspection requires the matching ReLU native graph".into());
    }
    let output = output.map(Path::to_path_buf).unwrap_or_else(|| {
        checkpoint
            .parent()
            .unwrap_or(Path::new("."))
            .join("feedback_depth_inspection.json")
    });
    if output == checkpoint
        || (output.exists() && output.canonicalize()? == checkpoint.canonicalize()?)
    {
        return Err("inspection output must be different from the model checkpoint".into());
    }
    let checkpoint_sha = reference_validation::sha256(checkpoint)?;
    let (recorded, verified_replay_record) =
        super::reference_eval::replay_expectation(checkpoint, replay_record)?;
    let tokens = TokenDataLoader::fineweb_validation_tokens()?;
    let bytes = super::data::validation_target_byte_count(&tokens)?;
    let mut trainer = Trainer::from_checkpoint(0x47505432, checkpoint)?;
    if trainer.latent_feedback.is_none() {
        return Err("checkpoint does not contain the latent-feedback graph".into());
    }
    let batch = trainer.batch_from_windows(&tokens, super::data::VALIDATION_WINDOWS)?;
    let before = trainer.eval_token_losses(&batch)?;
    let native_ce = before.iter().sum::<f32>() / before.len() as f32;
    if before.iter().any(|x| !x.is_finite())
        || recorded.is_some_and(|x| (x - f64::from(native_ce)).abs() > 1e-4)
    {
        return Err(
            "default P4 checkpoint replay does not match the recorded native endpoint".into(),
        );
    }
    let default_passes = super::latent_feedback::EVAL_PASSES;
    if default_passes != 4 {
        return Err("this diagnostic expects the established P4 default endpoint".into());
    }
    let mut depths = Vec::new();
    let mut first_positions = Vec::new();
    let mut finite = true;
    let mut first_position_invariant = true;
    let mut p4_exact = false;
    for passes in [1, 2, 3, 4] {
        let losses = trainer.eval_token_losses_for_passes(&batch, passes)?;
        let all_finite = losses.iter().all(|x| x.is_finite());
        finite &= all_finite;
        for sequence in 0..batch.batch_size {
            let value = losses[sequence * batch.seq_len].to_bits();
            if passes == 1 {
                first_positions.push(value);
            } else {
                first_position_invariant &= first_positions[sequence] == value;
            }
        }
        if passes == 4 {
            p4_exact = losses == before;
        }
        let ce = losses.iter().sum::<f32>() / losses.len() as f32;
        let bpb = f64::from(ce) * losses.len() as f64 / (bytes as f64 * std::f64::consts::LN_2);
        let row = json!({"passes":passes,"ce":ce,"bpb":bpb,"all_losses_finite":all_finite,
            "target_tokens":losses.len(),"target_bytes":bytes,"recorded_default_depth":passes==default_passes});
        println!("feedback_depth_inspection={row}");
        depths.push(row);
    }
    let after = trainer.eval_token_losses(&batch)?;
    p4_exact &= after == before;
    let unchanged = trainer.buffers.optimizer_state.step == 0;
    let passed = finite && p4_exact && first_position_invariant && unchanged;
    let report = json!({"passed":passed,"checkpoint":checkpoint,"checkpoint_sha256":checkpoint_sha,
        "default_eval_passes":default_passes,"default_native_ce":native_ce,"producer_recorded_native_ce":recorded,
        "verified_replay_provenance":verified_replay_record,"default_endpoint_replayed_bit_exact":p4_exact,
        "all_losses_finite":finite,"first_sequence_positions_identical":first_position_invariant,
        "batch_size":batch.batch_size,"context_tokens":batch.seq_len,"target_tokens":before.len(),"target_bytes":bytes,
        "optimizer_updates":trainer.buffers.optimizer_state.step,"depths":depths,
        "scope":"Same fixed saved weights and native held-out token/byte prefix; inference-depth diagnostic only. This does not replace the recorded P4 endpoint or full identical-text evaluation and is not a new training trajectory.",
    });
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!(
        "feedback_depth_inspection_result={} passed={passed}",
        output.display()
    );
    if !passed {
        return Err("feedback depth inspection failed; native results saved".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "completed current30k feedback checkpoint, actual native1B P1/P2/P3/P4; requires a free GPU"]
    fn native_completed_feedback_checkpoint_depths_replay_default_endpoint() -> crate::AppResult {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let checkpoint = root.join("target/runs/20261115_134812Z_fineweb_43200s/model.ckpt");
        let output = root.join("target/checks/20261115/feedback-depth-inspection.json");
        super::run(&checkpoint, Some(&output), None)?;
        let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&output)?)?;
        assert_eq!(report["passed"], true);
        assert_eq!(report["optimizer_updates"], 0);
        assert_eq!(report["default_eval_passes"], 4);
        assert_eq!(report["default_endpoint_replayed_bit_exact"], true);
        assert_eq!(report["depths"].as_array().unwrap().len(), 4);
        assert_eq!(report["target_tokens"], 8192);
        assert_eq!(report["target_bytes"], 34691);
        println!(
            "completed_feedback_depths_verified same_checkpoint=true passes=1,2,3,4 targets=8192 bytes=34691 P4_replay=bit_exact updates=0"
        );
        Ok(())
    }
}
