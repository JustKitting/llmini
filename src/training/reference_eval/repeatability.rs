use super::Trainer;
use crate::AppResult;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

pub(super) fn verify(
    trainer: &mut Trainer,
    validation: &[u16],
    baseline: &[f32],
) -> AppResult<serde_json::Value> {
    let native_windows = validation.len() / (GPT2_SEQ_LEN + 1);
    if native_windows == 0
        || native_windows > GPT2_BATCH_SIZE
        || GPT2_BATCH_SIZE % native_windows != 0
    {
        return Err("invalid validation shape for repeatability verification".into());
    }
    let native = trainer.batch_from_windows(validation, native_windows)?;
    let expanded = validation
        .iter()
        .copied()
        .cycle()
        .take(GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1))
        .collect::<Vec<_>>();
    let full = trainer.batch_from_windows(&expanded, GPT2_BATCH_SIZE)?;
    let mut maximum = 0_f32;
    let mut first_failure = None;
    let mut all_finite = true;
    for repetition in 0..128 {
        let batch = if repetition % 2 == 0 { &native } else { &full };
        let losses = trainer.eval_token_losses(batch)?;
        for (index, (&expected, &actual)) in baseline.iter().zip(&losses).enumerate() {
            let difference = (expected - actual).abs();
            all_finite &= actual.is_finite();
            if actual.is_finite() {
                maximum = maximum.max(difference);
            }
            if first_failure.is_none() && (!actual.is_finite() || difference > 1e-5) {
                first_failure = Some(serde_json::json!({
                    "repetition": repetition, "batch_size": batch.batch_size,
                    "target_index": index, "baseline_loss": expected,
                    "actual_loss": actual.to_string(), "absolute_difference": difference,
                }));
            }
        }
    }
    let passed = first_failure.is_none() && trainer.buffers.optimizer_state.step == 0;
    Ok(serde_json::json!({
        "passed": passed, "all_losses_finite": all_finite,
        "repeated_forwards": 128, "native_batch_size": native_windows,
        "expanded_batch_size": GPT2_BATCH_SIZE, "maximum_target_loss_difference": maximum,
        "first_failure": first_failure, "optimizer_updates": trainer.buffers.optimizer_state.step,
    }))
}
