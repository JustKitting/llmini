use std::{fs::File, io::BufReader, path::Path, time::Instant};

use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, Gpt2Rng};
use rust_kernels_cuda::hnet_stage::RatioObjective;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    Trainer,
    bytes::ByteDocuments,
    hierarchy::HierarchyScratch,
    parameters::{OUTER_WIDTH, Parameters, WeightViews},
    snapshot::{BACKBONE_FILE, OUTER_FILE, Snapshot},
};
use crate::{AppResult, reference_validation, training::launch::TrainConfig};

const OUTER_LAYERS: usize = 4;
const RATIO: RatioObjective = RatioObjective {
    target: 4.0,
    weight: 0.03,
};

pub(in crate::training) fn run(
    training: &Path,
    holdout: &Path,
    output: &Path,
    config: TrainConfig,
) -> AppResult {
    if training.canonicalize()? == holdout.canonicalize()? {
        return Err("H-Net training and holdout must use separate byte documents".into());
    }
    let mut documents = open(training)?;
    open(holdout)?;
    let batch_size = GPT2_BATCH_SIZE as u32;
    let stride = GPT2_SEQ_LEN as u32;
    let mut pending = documents.next_batch(batch_size, stride)?;
    if pending.is_none() {
        return Err("the H-Net training file has no byte targets".into());
    }
    let mut trainer = Trainer::new_full_kda(config.seed)?;
    let mut parameters = Parameters::new(
        &mut trainer,
        OUTER_LAYERS,
        RATIO.target,
        config.seed ^ 0x485f_4e45,
    )?;
    let mut scratch =
        HierarchyScratch::new(&trainer.runtime, batch_size, stride, OUTER_WIDTH as u32)?;
    let mut rng = Gpt2Rng::new(config.seed ^ 0x5f4e_4554);
    println!("run_dir={}", output.display());
    println!(
        "hnet_training_bytes={} hnet_holdout_bytes={} batch={} sequence_bytes={} outer_width={} encoder_layers={} decoder_layers={} ratio={} seed={} steps={}",
        training.display(),
        holdout.display(),
        batch_size,
        stride,
        OUTER_WIDTH,
        OUTER_LAYERS,
        OUTER_LAYERS,
        RATIO.target,
        config.seed,
        config.step_cap
    );

    let mut elapsed = 0.0;
    let mut attempts = 0;
    let mut trained_bytes = 0_u64;
    let stop_on_nonfinite = crate::training::env::env_bool("TRAIN_STOP_ON_NONFINITE").unwrap_or(false);
    for index in 0..config.step_cap {
        if elapsed >= config.max_seconds {
            break;
        }
        let start = Instant::now();
        // Each reopening preserves the complete document boundaries at the epoch edge.
        let batch = if let Some(batch) = pending.take() {
            batch
        } else if let Some(batch) = documents.next_batch(batch_size, stride)? {
            batch
        } else {
            documents = open(training)?;
            documents
                .next_batch(batch_size, stride)?
                .ok_or("H-Net training file stopped supplying targets")?
        };
        let step =
            trainer.train_hnet_step(&mut scratch, &mut parameters, &batch, RATIO, &mut rng)?;
        trainer.runtime.stream.synchronize()?;
        let step_time = start.elapsed().as_secs_f64();
        elapsed += step_time;
        attempts = index + 1;
        trained_bytes += u64::from(batch.raw_bytes);
        let optimizer = step.optimizer;
        let nonfinite = step.has_nonfinite();
        if index.is_multiple_of(config.log_interval)
            || attempts == config.step_cap
            || optimizer.update_skipped
            || nonfinite
        {
            println!(
                "hnet_step attempt={attempts} optimizer_step={} byte_ce={:.8} ratio_loss={:.8} grad_norm={:.6} skipped={} skip_loss_spike={} skip_grad_norm_spike={} skip_non_finite={} step_ms={:.3} trained_bytes={trained_bytes}",
                trainer.buffers.optimizer_state.step,
                step.byte_ce,
                step.ratio_loss,
                optimizer.grad_norm,
                optimizer.update_skipped,
                optimizer.skip_loss_spike,
                optimizer.skip_grad_norm_spike,
                optimizer.skip_non_finite,
                step_time * 1000.0
            );
        }
        if stop_on_nonfinite && nonfinite {
            return Err(format!("nonfinite H-Net training at attempted step {attempts}").into());
        }
        if config
            .eval_interval
            .is_some_and(|interval| attempts.is_multiple_of(interval))
            && attempts < config.step_cap
            && elapsed < config.max_seconds
        {
            let metrics = score_holdout(&mut trainer, &mut parameters, &mut scratch, holdout)?;
            println!("hnet_holdout attempt={attempts} {metrics}");
        }
    }

    let metrics = score_holdout(&mut trainer, &mut parameters, &mut scratch, holdout)?;
    let updates = trainer.buffers.optimizer_state.step;
    println!(
        "hnet_holdout attempt={attempts} completed_steps={updates} train_seconds={elapsed:.3} {metrics}"
    );
    let report = json!({
        "training_documents": training, "holdout_documents": holdout, "seed": config.seed,
        "batch_size": batch_size, "context_bytes": stride, "encoder_layers": OUTER_LAYERS,
        "decoder_layers": OUTER_LAYERS, "outer_width": OUTER_WIDTH, "target_ratio": RATIO.target,
        "ratio_loss_weight": RATIO.weight, "attempts": attempts, "completed_steps": updates,
        "trained_bytes": trained_bytes, "train_seconds": elapsed, "holdout": metrics,
    });
    let backbone = output.join("hnet-backbone.ckpt.partial");
    let outer = output.join("hnet-outer.ckpt.partial");
    trainer.save_model(&backbone)?;
    parameters.save_weights(&trainer.runtime.stream, &outer)?;
    std::fs::rename(backbone, output.join(BACKBONE_FILE))?;
    std::fs::rename(outer, output.join(OUTER_FILE))?;
    let temporary = output.join("holdout.json.partial");
    let metrics_output = File::create_new(&temporary)?;
    serde_json::to_writer_pretty(metrics_output, &report)?;
    std::fs::rename(temporary, output.join("holdout.json"))?;
    Ok(())
}

fn open(path: &Path) -> AppResult<ByteDocuments<BufReader<File>>> {
    Ok(ByteDocuments::new(BufReader::new(File::open(path)?))?)
}

fn score_holdout(
    trainer: &mut Trainer,
    parameters: &mut Parameters,
    scratch: &mut HierarchyScratch,
    path: &Path,
) -> AppResult<serde_json::Value> {
    trainer.materialize_evaluation_weights()?;
    parameters.materialize(&trainer.runtime, &mut trainer.buffers.optimizer, 1.0)?;
    score_averaged(trainer, scratch, path, parameters.weights())
}

pub(in crate::training) fn evaluate(
    run: &Path,
    holdout: &Path,
    output: Option<&Path>,
) -> AppResult {
    let backbone = run.join(BACKBONE_FILE);
    let outer = run.join(OUTER_FILE);
    let backbone_sha = crate::reference_validation::sha256(&backbone)?;
    let outer_sha = crate::reference_validation::sha256(&outer)?;
    open(holdout)?;
    let mut trainer = Trainer::from_hnet_backbone_checkpoint(0x47505432, File::open(&backbone)?)?;
    let snapshot = Snapshot::load(&trainer.runtime, &outer, OUTER_LAYERS)?;
    let mut scratch = HierarchyScratch::new(
        &trainer.runtime,
        GPT2_BATCH_SIZE as u32,
        GPT2_SEQ_LEN as u32,
        OUTER_WIDTH as u32,
    )?;
    let metrics = score_averaged(&mut trainer, &mut scratch, holdout, snapshot.weights())?;
    let report = json!({
        "backbone_checkpoint": backbone,
        "backbone_sha256": backbone_sha,
        "outer_checkpoint": outer,
        "outer_sha256": outer_sha,
        "holdout_documents": holdout,
        "context_bytes": GPT2_SEQ_LEN,
        "batch_size": GPT2_BATCH_SIZE,
        "holdout": metrics,
    });
    let text = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        use std::io::Write;
        let mut file = File::create_new(path)?;
        writeln!(file, "{text}")?;
    }
    println!("{text}");
    Ok(())
}

fn score_averaged(
    trainer: &mut Trainer,
    scratch: &mut HierarchyScratch,
    path: &Path,
    weights: WeightViews<'_>,
) -> AppResult<serde_json::Value> {
    let mut documents = open(path)?;
    let mut bytes = 0_u64;
    let mut targets = 0_u64;
    let mut nll = 0.0_f64;
    let mut digest = Sha256::new();
    let mut raw = Vec::new();
    let continuation_stride =
        (GPT2_SEQ_LEN * reference_validation::STRIDE / reference_validation::CONTEXT) as u32;
    while let Some(batch) = documents.next_holdout_batch(
        GPT2_BATCH_SIZE as u32,
        GPT2_SEQ_LEN as u32,
        continuation_stride,
    )? {
        let losses = trainer.score_hnet_bytes(
            scratch,
            &batch,
            weights.embedding,
            weights.hierarchy(),
            RATIO,
        )?;
        nll += losses.into_iter().map(f64::from).sum::<f64>();
        bytes += u64::from(batch.raw_bytes);
        targets += u64::from(batch.scored_targets);
        raw.clear();
        raw.extend(batch.targets.iter().filter_map(|&id| u8::try_from(id).ok()));
        digest.update(&raw);
    }
    if bytes == 0 || !nll.is_finite() {
        return Err("the H-Net holdout is empty or has a nonfinite score".into());
    }
    Ok(json!({
        "byte_ce": nll / targets as f64,
        "bits_per_byte": nll / (std::f64::consts::LN_2 * bytes as f64),
        "scored_targets": targets, "raw_bytes": bytes,
        "continuation_stride_bytes": continuation_stride,
        "raw_sha256": format!("{:x}", digest.finalize()),
    }))
}
