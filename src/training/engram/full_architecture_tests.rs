//! Native full-graph contract, declared before the multi-layer implementation.
use crate::{
    AppResult,
    training::{
        Trainer,
        optimizer_apply::{WeightUpdateArgs, apply_weight_updates},
    },
};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use std::path::PathBuf;

#[test]
#[ignore = "native full-size reference Engram graph and state lifecycle"]
fn native_engram_full_reference_graph_and_lifecycle() -> AppResult {
    let mut t = Trainer::new(0x47505432)?;
    {
        let e = t.engram.as_ref().unwrap();
        let named = e.named_gradients();
        assert_eq!(
            named.len(),
            30,
            "two full modules must expose every matrix, vector and learned control"
        );
        for layer in [1, 5] {
            let prefix = format!("engram.layers.{layer}.");
            let table = named
                .iter()
                .find(|(n, _)| *n == format!("{prefix}table"))
                .expect("missing full reference memory layer");
            assert!(
                table.1.len() > 160_000_000,
                "full reference prime capacity/head dimension"
            );
            for name in ["key.weight", "value.weight"] {
                let g = named
                    .iter()
                    .find(|(n, _)| *n == format!("{prefix}{name}"))
                    .unwrap()
                    .1;
                assert_eq!(g.len(), 1024 * GPT2_N_EMBD);
                assert!(
                    t.buffers
                        .muon_tables
                        .matrix_grad_ranges()
                        .iter()
                        .any(|(p, n)| *p == g.cu_deviceptr() && *n as usize == g.len())
                );
            }
        }
    }
    let windows = (0..GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1))
        .map(|i| {
            if i % (GPT2_SEQ_LEN + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let b = t.batch_from_windows(&windows, GPT2_BATCH_SIZE)?;
    for _ in 0..2 {
        let s = t.train_step(&b, true)?;
        assert!(s.finite && !s.optimizer.update_skipped && s.optimizer.grad_norm.is_finite());
    }
    t.materialize_training_weights()?;
    t.forward_step(&b)?;
    t.enqueue_numeric_probe_backward(&b)?;
    let named = t.engram.as_ref().unwrap().named_gradients();
    for (name, g) in &named {
        let a = g.to_host_vec(&t.runtime.stream)?;
        assert!(a.iter().all(|x| x.is_finite()), "nonfinite {name}");
        assert!(a.iter().any(|x| *x != 0.), "gradient-dead {name}");
    }
    let components = t.buffers.grad_clip.norm_components(
        &t.runtime.stream,
        &t.buffers.backward,
        &t.buffers.next_latent_grads,
        &t.buffers.muon_tables,
    )?;
    for layer in [1, 5] {
        assert_eq!(
            components
                .iter()
                .filter(|c| c["name"]
                    .as_str()
                    .unwrap()
                    .starts_with(&format!("engram.layers.{layer}.")))
                .count(),
            15
        );
    }
    let bad = named
        .iter()
        .find(|(n, _)| *n == "engram.layers.5.table")
        .unwrap()
        .1;
    unsafe {
        cuda_core::memory::memset_d8_async(
            bad.cu_deviceptr(),
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    let step = t.buffers.optimizer_state.step;
    let result = apply_weight_updates(WeightUpdateArgs {
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
    assert!(result.trace.update_skipped && result.trace.skip_non_finite);
    assert_eq!(step, t.buffers.optimizer_state.step);
    t.materialize_evaluation_weights()?;
    let loss = t.eval_token_losses(&b)?;
    assert_eq!(loss, t.eval_token_losses(&b)?);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/engram-full-reference-roundtrip.ckpt");
    std::fs::create_dir_all(path.parent().unwrap())?;
    t.save_model(&path)?;
    t.load_model(&path)?;
    assert_eq!(t.engram.as_ref().unwrap().named_gradients().len(), 30);
    assert_eq!(loss, t.eval_token_losses(&b)?);
    let mut changed = windows.clone();
    for seq in 0..GPT2_BATCH_SIZE {
        for x in &mut changed[seq * (GPT2_SEQ_LEN + 1) + 1024..(seq + 1) * (GPT2_SEQ_LEN + 1)] {
            *x = 28001;
        }
    }
    let b2 = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
    let altered = t.eval_token_losses(&b2)?;
    for seq in 0..GPT2_BATCH_SIZE {
        let at = seq * GPT2_SEQ_LEN;
        assert_eq!(&loss[at..at + 1023], &altered[at..at + 1023]);
    }
    println!(
        "full_reference_engram passed=true modules=2 width=1024 trainable_groups=30 common_muon_matrices=4 second_module_nonfinite_global_rejection=true checkpoint_replay=true causality=true fixture_updates=2"
    );
    Ok(())
}
