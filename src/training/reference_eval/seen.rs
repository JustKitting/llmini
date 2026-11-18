//! Frozen-checkpoint fit on a fixed sample of the run's observed training batches.
use super::Trainer;
use crate::AppResult;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use llama2_tokenizer::TrainingTokenizer;
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub(super) fn score(
    trainer: &mut Trainer,
    checkpoint: &Path,
) -> AppResult<Option<serde_json::Value>> {
    let Some(parent) = checkpoint.parent() else {
        return Ok(None);
    };
    let metrics = parent.join("metrics/train/epoch-1");
    let sources_path = metrics.join("Source.log");
    let offsets_path = metrics.join("Window_offset.log");
    if !sources_path.exists() && !offsets_path.exists() {
        return Ok(None);
    }
    let sources_text = fs::read_to_string(sources_path)?;
    let offsets_text = fs::read_to_string(offsets_path)?;
    let sources = sources_text.lines().collect::<Vec<_>>();
    let offsets = offsets_text.lines().collect::<Vec<_>>();
    if sources.is_empty() || sources.len() != offsets.len() {
        return Err("seen-data replay source/offset metric lengths differ".into());
    }
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let sample_count = 32.min(sources.len());
    let span = GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1;
    let mut total_nats = 0_f64;
    let mut total_tokens = 0_usize;
    let mut total_bytes = 0_usize;
    let mut samples = Vec::new();
    for sample in 0..sample_count {
        // Midpoints of equal strata; independent of losses and gradient norms.
        let row = ((2 * sample + 1) * sources.len()) / (2 * sample_count);
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(sources[row]);
        let offset: u64 = offsets[row].parse()?;
        let mut input = File::open(&source)?;
        input.seek(SeekFrom::Start(
            offset.checked_mul(2).ok_or("token offset overflow")?,
        ))?;
        let mut raw = vec![0_u8; span * 2];
        input.read_exact(&mut raw)?;
        let contiguous = raw
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect::<Vec<_>>();
        if contiguous
            .iter()
            .any(|&v| usize::from(v) >= llama2_tokenizer::MODEL_VOCAB_SIZE)
        {
            return Err("seen-data replay token is outside the model vocabulary".into());
        }
        let mut flat = Vec::with_capacity(GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1));
        let mut bytes = 0;
        for sequence in 0..GPT2_BATCH_SIZE {
            let start = sequence * GPT2_SEQ_LEN;
            flat.extend_from_slice(&contiguous[start..start + GPT2_SEQ_LEN + 1]);
            let target_ids = contiguous[start + 1..start + GPT2_SEQ_LEN + 1]
                .iter()
                .map(|&v| u32::from(v))
                .collect::<Vec<_>>();
            bytes += tokenizer.decode(&target_ids)?.len();
        }
        if bytes == 0 {
            return Err("seen-data replay decoded to zero target bytes".into());
        }
        let batch = trainer.batch_from_windows(&flat, GPT2_BATCH_SIZE)?;
        let losses = trainer.eval_token_losses(&batch)?;
        if losses.len() != GPT2_BATCH_SIZE * GPT2_SEQ_LEN || losses.iter().any(|v| !v.is_finite()) {
            return Err(
                "seen-data replay has nonfinite losses or an unexpected target count".into(),
            );
        }
        let nats = losses.iter().map(|&v| f64::from(v)).sum::<f64>();
        total_nats += nats;
        total_tokens += losses.len();
        total_bytes += bytes;
        samples.push(serde_json::json!({
            "logged_metric_row": row + 1,
            "source": sources[row],
            "token_offset": offset,
            "target_tokens": losses.len(),
            "target_bytes": bytes,
            "mean_token_ce": nats / losses.len() as f64,
        }));
    }
    if trainer.buffers.optimizer_state.step != 0 {
        return Err("seen-data replay changed optimizer state".into());
    }
    let ce = total_nats / total_tokens as f64;
    let bpb = total_nats / total_bytes as f64 / std::f64::consts::LN_2;
    println!(
        "seen_data_replay batches={sample_count} targets={total_tokens} mean_token_ce={ce:.9} bits_per_byte={bpb:.9}"
    );
    Ok(Some(serde_json::json!({
        "selection": "midpoint of each of 32 equal strata of saved metric rows, or all rows when fewer than 32",
        "frozen_evaluation_checkpoint": true,
        "optimizer_updates": 0,
        "batches": sample_count,
        "target_tokens": total_tokens,
        "target_bytes": total_bytes,
        "loss_sum_nats": total_nats,
        "mean_token_ce": ce,
        "bits_per_byte": bpb,
        "samples": samples,
        "interpretation": "Batches recorded by this training run, scored with the final averaged checkpoint.",
    })))
}
