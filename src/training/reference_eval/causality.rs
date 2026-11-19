use super::{Trainer, input::Window};
use crate::AppResult;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

pub(super) fn verify(trainer: &mut Trainer, windows: &[Window]) -> AppResult<serde_json::Value> {
    let window_len = GPT2_SEQ_LEN + 1;
    let selected = &windows[..GPT2_BATCH_SIZE];
    let flat = selected
        .iter()
        .flat_map(|w| w.ids.iter().copied())
        .collect::<Vec<_>>();
    let batch = trainer.batch_from_windows(&flat, GPT2_BATCH_SIZE)?;
    let baseline = trainer.eval_token_losses(&batch)?;
    let repeated = trainer.eval_token_losses(&batch)?;
    let repeat_max = max_difference(&baseline, &repeated)?;
    let mut changed_padding = flat.clone();
    for (sequence, window) in selected.iter().enumerate() {
        changed_padding[sequence * window_len + window.hi + 1..(sequence + 1) * window_len].fill(2);
    }
    let padding_batch = trainer.batch_from_windows(&changed_padding, GPT2_BATCH_SIZE)?;
    let padding_losses = trainer.eval_token_losses(&padding_batch)?;
    let mut padding_max = 0_f32;
    for (sequence, window) in selected.iter().enumerate() {
        let start = sequence * GPT2_SEQ_LEN + window.lo;
        let end = sequence * GPT2_SEQ_LEN + window.hi;
        padding_max = padding_max.max(max_difference(
            &baseline[start..end],
            &padding_losses[start..end],
        )?);
    }

    let mut prefix_max = 0_f32;
    for cut in [1, 64, 127, 128, 166, 255, 511, 1023, 2047] {
        let mut changed = flat.clone();
        for window in changed.chunks_exact_mut(window_len) {
            for (index, token) in window.iter_mut().enumerate().skip(cut) {
                *token = if index % 2 == 0 { 22557 } else { 28808 };
            }
        }
        let mut changed_batch = trainer.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
        changed_batch.targets = trainer.batch_from_windows(&flat, GPT2_BATCH_SIZE)?.targets;
        let changed_losses = trainer.eval_token_losses(&changed_batch)?;
        let mut difference = 0_f32;
        for sequence in 0..GPT2_BATCH_SIZE {
            let start = sequence * GPT2_SEQ_LEN;
            difference = difference.max(max_difference(
                &baseline[start..start + cut],
                &changed_losses[start..start + cut],
            )?);
        }
        prefix_max = prefix_max.max(difference);
        println!("reference_causality_prefix cut={cut} maximum_difference={difference:.9}");
    }

    let mut changed = flat.clone();
    changed[window_len..].fill(22557);
    let changed_batch = trainer.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
    let changed_losses = trainer.eval_token_losses(&changed_batch)?;
    let sequence_max = max_difference(&baseline[..GPT2_SEQ_LEN], &changed_losses[..GPT2_SEQ_LEN])?;
    if repeat_max > 1e-5 || padding_max > 1e-5 || prefix_max > 1e-5 || sequence_max > 1e-5 {
        return Err(format!("reference causality failure: repeat={repeat_max}, padding={padding_max}, prefix={prefix_max}, sequence={sequence_max}").into());
    }
    Ok(serde_json::json!({
        "batch_size": GPT2_BATCH_SIZE,
        "prefix_cases": 9,
        "repeat_max_difference": repeat_max,
        "future_padding_max_difference": padding_max,
        "future_prefix_max_difference": prefix_max,
        "cross_sequence_max_difference": sequence_max,
    }))
}

fn max_difference(a: &[f32], b: &[f32]) -> AppResult<f32> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("nonfinite or mismatched loss vector in causality verification".into());
    }
    Ok(a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max))
}
