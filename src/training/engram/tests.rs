//! Native full-model integration gate; not a convergence or loss result.
use super::*;
use crate::training::{
    Trainer,
    optimizer_apply::{WeightUpdateArgs, apply_weight_updates},
};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use std::path::PathBuf;
#[test]
#[ignore = "full approximately 1B native CUDA integration on GPU 0"]
fn native_engram_gradients_clipping_checkpoint_and_causality() -> crate::AppResult {
    let mut t = Trainer::new(0x47505432)?;
    assert!(t.engram.is_some());
    let projector_gradients = &t.engram.as_ref().unwrap().modules[0].ng;
    let matrix_ranges = t.buffers.muon_tables.matrix_grad_ranges();
    for index in [1, 3] {
        assert!(
            matrix_ranges.iter().any(|(ptr, len)| *ptr
                == projector_gradients[index].cu_deviceptr()
                && *len as usize == MEMORY * gpt2_nvfp4::GPT2_N_EMBD),
            "Engram dense projector {index} missing from the native Muon/SymExpLin matrix table"
        );
    }
    let map = t.uploaded.engram.as_ref().unwrap().modules[0]
        .token_map
        .to_host_vec(&t.runtime.stream)?;
    assert_eq!(map[16230], map[21558]); // Hello / hello
    assert_eq!(map[16230], map[22557]); // leading-space spelling
    assert_eq!(map[28797], map[28901]); // accented lowercase / uppercase
    assert_ne!(map[0], map[1]);
    assert_ne!(map[1], map[2]);
    assert_ne!(map[3], map[4]); // byte fallback IDs stay distinct
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
        let st = t.train_step(&b, true)?;
        assert!(st.finite && !st.optimizer.update_skipped && st.optimizer.grad_norm.is_finite());
    }
    t.materialize_training_weights()?;
    t.forward_step(&b)?;
    t.enqueue_numeric_probe_backward(&b)?;
    let e = &t.engram.as_ref().unwrap().modules[0];
    let mut norms = Vec::new();
    for (i, g) in e.gradients().iter().enumerate() {
        let a = g.to_host_vec(&t.runtime.stream)?;
        assert!(
            a.iter().all(|x| x.is_finite()),
            "nonfinite Engram gradient {i}"
        );
        let norm = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        assert!(norm > 0., "inactive Engram gradient {i}");
        norms.push(norm);
    }
    let before_step = t.buffers.optimizer_state.step;
    let table_before = e.nv_states[0].z_master.to_host_vec(&t.runtime.stream)?;
    unsafe {
        cuda_core::memory::memset_d8_async(
            e.ng[0].cu_deviceptr(),
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
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
    let failure_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261114/engram-forced-failure.tsv");
    std::fs::create_dir_all(failure_path.parent().unwrap())?;
    t.write_numerical_failure_dump(&b, &failure_path)?;
    let failure = std::fs::read_to_string(&failure_path)?;
    assert!(failure.contains(&format!(
        "engram.layers.1.table.gradient\t{}\t1\t0\t",
        TABLE_LEN
    )));
    assert!(failure.contains("engram.layers.1.convolution.z\t"));
    assert_eq!(before_step, t.buffers.optimizer_state.step);
    assert_eq!(
        table_before,
        t.engram.as_ref().unwrap().modules[0].nv_states[0]
            .z_master
            .to_host_vec(&t.runtime.stream)?
    );
    t.materialize_evaluation_weights()?;
    let loss = t.eval_token_losses(&b)?;
    let repeat = t.eval_token_losses(&b)?;
    assert_eq!(loss, repeat);
    let checkpoint = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261114/engram-native-roundtrip.ckpt");
    t.save_model(&checkpoint)?;
    t.load_model(&checkpoint)?;
    let replay = t.eval_token_losses(&b)?;
    assert_eq!(loss, replay);
    let mut changed = windows.clone();
    for seq in 0..GPT2_BATCH_SIZE {
        let base = seq * (GPT2_SEQ_LEN + 1);
        for x in &mut changed[base + 1024..base + GPT2_SEQ_LEN + 1] {
            *x = 28001;
        }
    }
    let changed = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
    let altered = t.eval_token_losses(&changed)?;
    for seq in 0..GPT2_BATCH_SIZE {
        assert_eq!(
            &loss[seq * GPT2_SEQ_LEN..seq * GPT2_SEQ_LEN + 1023],
            &altered[seq * GPT2_SEQ_LEN..seq * GPT2_SEQ_LEN + 1023]
        );
    }
    let report = serde_json::json!({"passed":true,"model":"native approximately 1B, intact NextLat","batch":GPT2_BATCH_SIZE,"context":GPT2_SEQ_LEN,"engram_gradient_norms":norms,"projectors_in_native_muon_table":true,"engram_parameter_groups":15,"forced_nonfinite_skipped":true,"state_advance_on_skip":false,"checkpoint_loss_bit_exact":true,"forward_repeat_bit_exact":true,"future_prefix_loss_bit_exact":true,"training_updates":2,"checkpoint":checkpoint});
    std::fs::write(
        checkpoint.with_extension("verification.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
