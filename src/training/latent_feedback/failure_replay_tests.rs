//! The exact captured 15559 failure: model/input/pass count, zero updates.
use crate::{AppResult, training::Trainer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

#[test]
#[ignore = "native B8 replay of causal-feedback attempt15559, captured frozen weights/input, no optimizer replay"]
fn native_feedback_failure_15559_layernorm_tape_and_backward() -> AppResult {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let checkpoint =
        root.join("target/runs/20261115_111114Z_fineweb_43200s/numerical_failure_weights.ckpt");
    crate::reference_validation::require_hash(
        &checkpoint,
        "7233bcd236d91f6b1e9a6c1ef64689635d0745a65fb16b433a5816cbb4033f11",
    )?;
    assert_eq!((GPT2_BATCH_SIZE, GPT2_SEQ_LEN), (8, 2048));
    let path = root.join("target/checks/20261115/feedback-failure-15559-input.u16");
    crate::reference_validation::require_hash(
        &path,
        "c41df62b74ab4c4054df75afb09503ed40b54daae0d64625a469c701eb7939a3",
    )?;
    let raw = std::fs::read(&path)?;
    assert_eq!(raw.len(), (GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1) * 2);
    let contiguous = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<_>>();
    let mut windows = Vec::new();
    for i in 0..GPT2_BATCH_SIZE {
        windows
            .extend_from_slice(&contiguous[i * GPT2_SEQ_LEN..i * GPT2_SEQ_LEN + GPT2_SEQ_LEN + 1]);
    }
    let mut t = Trainer::new_feedback(0x47505432)?;
    t.load_model(&checkpoint)?;
    let batch = t.batch_from_windows(&windows, GPT2_BATCH_SIZE)?;
    let losses = t.eval_token_losses_for_passes(&batch, 2)?;
    assert!(losses.iter().all(|x| x.is_finite()));
    let tape = t.buffers.tape.saved(
        &batch.tokens,
        8,
        2048,
        batch.token_count as u32,
        &t.buffers.logits,
    );
    let mut bad = Vec::new();
    for (index, b) in tape.blocks.iter().enumerate() {
        for (norm, saved) in [(1, &b.ln_1), (2, &b.ln_2)] {
            let nonfinite = saved
                .residual
                .to_host_vec(&t.runtime.stream)?
                .into_iter()
                .filter(|x| x & 0x7c00 == 0x7c00)
                .count();
            if nonfinite > 0 {
                bad.push((index, norm, nonfinite));
            }
        }
    }
    t.enqueue_numeric_probe_backward(&batch)?;
    let gradient = t
        .latent_feedback
        .as_ref()
        .unwrap()
        .grad
        .to_host_vec(&t.runtime.stream)?;
    let bad_gradient = gradient.iter().filter(|x| !x.is_finite()).count();
    let global = t
        .buffers
        .grad_clip
        .clip(&t.runtime.stream, &t.runtime.optimizer)?;
    println!(
        "feedback_failure_15559_replay primary_ce={} nonfinite_norm_tapes={bad:?} nonfinite_fusion_gradient={bad_gradient} passes=2 source=train_000013 offset=5046272 updates=0 optimizer_state_replayed=false",
        losses.iter().map(|&x| f64::from(x)).sum::<f64>() / losses.len() as f64
    );
    assert!(
        bad.is_empty(),
        "finite residuals must not overflow normalization tapes"
    );
    assert_eq!(
        bad_gradient, 0,
        "the exact captured full backward must stay finite"
    );
    assert!(
        global.norm.is_finite(),
        "every parameter/control gradient must remain finite"
    );
    println!(
        "feedback_failure_15559_global_gradient_norm={} all_gradients_finite=true",
        global.norm
    );
    assert_eq!(t.buffers.optimizer_state.step, 0);
    Ok(())
}
