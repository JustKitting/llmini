//! Complete positional-component lifecycle cases declared before production changes.
use crate::{
    AppResult,
    training::{
        Trainer,
        optimizer_apply::{WeightUpdateArgs, apply_weight_updates},
    },
};
use gpt2_nvfp4::{GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D, GPT2_SEQ_LEN as S};

#[test]
#[ignore = "complete native B8/L16/d2048 stationary-position model, optimizer, checkpoint and causality"]
fn native_stationary_rope_model_gradients_optimizer_checkpoint_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_STATIONARY_ROPE);
    let mut t = Trainer::new_stationary_rope(0x47505432)?;
    assert!(t.engram.is_none() && t.mhc.is_none() && t.latent_feedback.is_none());
    let flat = (0..B * (S + 1))
        .map(|i| {
            if i % (S + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let b = t.batch_from_windows(&flat, B)?;
    for _ in 0..2 {
        let step = t.train_step(&b, true)?;
        assert!(
            step.finite && !step.optimizer.update_skipped && step.optimizer.grad_norm.is_finite()
        );
    }
    t.materialize_training_weights()?;
    t.forward_step(&b)?;
    t.enqueue_numeric_probe_backward(&b)?;
    let clip = t
        .buffers
        .grad_clip
        .clip(&t.runtime.stream, &t.runtime.optimizer)?;
    assert!(clip.norm.is_finite());
    let components = t.buffers.grad_clip.norm_components(
        &t.runtime.stream,
        &t.buffers.backward,
        &t.buffers.next_latent_grads,
        &t.buffers.muon_tables,
    )?;
    for layer in 0..16 {
        for suffix in [
            "attn_qkv.weight",
            "attn_c_proj.weight",
            "mlp_up.weight",
            "mlp_down.weight",
        ] {
            let name = format!("blocks.{layer}.{suffix}");
            let c = components
                .iter()
                .find(|c| c["name"].as_str() == Some(&name))
                .unwrap();
            assert!(
                c["finite"] == true && c["l2"].as_f64().unwrap() > 0.,
                "inactive {name}"
            );
        }
    }
    for suffix in ["input_projection", "transition", "output_projection"] {
        let name = format!("next_latent.{suffix}.weight");
        let c = components
            .iter()
            .find(|c| c["name"].as_str() == Some(&name))
            .unwrap();
        assert!(c["finite"] == true && c["l2"].as_f64().unwrap() > 0.);
    }
    let bad = &t.buffers.backward.blocks[15].d_attn_qkv_bias;
    unsafe {
        cuda_core::memory::memset_d8_async(
            bad.cu_deviceptr() + (2 * D * 4) as u64,
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    let step = t.buffers.optimizer_state.step;
    let rejected = apply_weight_updates(WeightUpdateArgs {
        stream: &t.runtime.stream,
        runtime: &t.runtime,
        batch: &b,
        uploaded: &mut t.uploaded,
        grads: &mut t.buffers.backward,
        next_latent_grads: &mut t.buffers.next_latent_grads,
        observed_loss: Some(5.),
        scratch: &mut t.buffers.optimizer,
        state: &mut t.buffers.optimizer_state,
        muon: &mut t.buffers.muon,
        muon_tables: &t.buffers.muon_tables,
        tape: &t.buffers.tape,
        qk_norm_max: t.buffers.scratch.attention_core.qk_norm_max(),
        grad_clip: &mut t.buffers.grad_clip,
    })?;
    assert!(rejected.trace.update_skipped && rejected.trace.skip_non_finite);
    assert_eq!(step, t.buffers.optimizer_state.step);
    t.materialize_evaluation_weights()?;
    let before = t.eval_token_losses(&b)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/stationary-rope-complete-roundtrip.ckpt");
    t.save_model(&path)?;
    t.load_model(&path)?;
    assert_eq!(before, t.eval_token_losses(&b)?);
    let legacy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/runs/20261114_080934Z_fineweb_21600s/model.ckpt");
    assert!(
        t.load_model(&legacy).is_err(),
        "legacy positional metadata silently reinterpreted"
    );
    assert_eq!(before, t.eval_token_losses(&b)?);
    for cut in [1, 31, 63, 64, 127, 128, 1023, 2015, 2047] {
        let mut changed = flat.clone();
        for window in changed.chunks_exact_mut(S + 1) {
            window[cut + 1..].fill(28001);
        }
        let changed = t.batch_from_windows(&changed, B)?;
        let loss = t.eval_token_losses(&changed)?;
        for seq in 0..B {
            let start = seq * S;
            assert_eq!(&before[start..start + cut], &loss[start..start + cut]);
        }
    }
    let mut changed = flat.clone();
    changed[..S + 1].fill(28002);
    let changed = t.batch_from_windows(&changed, B)?;
    assert_eq!(&before[S..], &t.eval_token_losses(&changed)?[S..]);
    println!(
        "complete_stationary_rope_model_verified B={B} S={S} layers=16 d={D} recurrent_layers=12 stationary_full_attention_layers=4 all_principal_base_and_NextLat_gradients=active global_nonfinite_skip=true averaged_checkpoint=bit_exact legacy_position_graph_rejected=true prefix_cases=9 cross_sequence=bit_exact fixture_updates=2"
    );
    Ok(())
}
