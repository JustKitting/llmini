//! Complete-model contract declared before trainer integration, not a loss screen.
use crate::{
    AppResult,
    training::{
        Trainer,
        optimizer_apply::{WeightUpdateArgs, apply_weight_updates},
    },
};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_SEQ_LEN};

#[test]
#[ignore = "native full mHC B8/L16/d2048 graph, all32 boundaries, readout and lifecycle"]
fn native_mhc_full_graph_gradients_checkpoint_and_causality() -> AppResult {
    let mut t = Trainer::new(0x47505432)?;
    assert!(
        t.engram.is_none() && t.latent_feedback.is_none(),
        "this gate is a separate full mHC trial"
    );
    let state = t
        .mhc
        .as_ref()
        .expect("TRAIN_MHC=1 must select the actual full graph");
    let named = state.named_gradients();
    let weights = t.uploaded.mhc.as_ref().unwrap();
    assert_eq!(weights.initialization_version, 2);
    for (i, unit) in weights.units.iter().enumerate() {
        let actual = unit.alpha.weight.to_host_vec(&t.runtime.stream)?;
        let expected = if i == 32 { vec![1.] } else { vec![0.01; 3] };
        assert_eq!(
            actual, expected,
            "paper branch gate initialization / library HyperHead initialization"
        );
    }
    assert_eq!(
        named.len(),
        33 * 6,
        "all32 dynamic boundaries and learned HyperHead must be active"
    );
    for boundary in 0..32 {
        let prefix = format!("mhc.boundaries.{boundary}.");
        assert_eq!(
            named.iter().filter(|(n, _)| n.starts_with(&prefix)).count(),
            6
        );
    }
    assert_eq!(
        named
            .iter()
            .filter(|(n, _)| n.starts_with("mhc.head."))
            .count(),
        6
    );
    let ranges = t.buffers.muon_tables.matrix_grad_ranges();
    for (name, g) in named
        .iter()
        .filter(|(n, _)| n.ends_with("projection.weight"))
    {
        assert_eq!(
            g.len(),
            128 * 4 * GPT2_N_EMBD,
            "physical NVFP4 padding, not extra model capacity"
        );
        assert!(
            ranges
                .iter()
                .any(|(ptr, len)| *ptr == g.cu_deviceptr() && *len as usize == g.len()),
            "missing common Muon matrix {name}"
        );
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
    let named = t.mhc.as_ref().unwrap().named_gradients();
    let mut measured = Vec::new();
    for (name, g) in &named {
        let a = g.to_host_vec(&t.runtime.stream)?;
        assert!(a.iter().all(|x| x.is_finite()), "nonfinite {name}");
        assert!(a.iter().any(|x| *x != 0.), "gradient-dead {name}");
        measured.push((
            name.to_string(),
            a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>(),
        ));
        if name.ends_with("projection.weight") {
            let active = if name.starts_with("mhc.head.") { 4 } else { 24 };
            assert!(
                a[active * 4 * GPT2_N_EMBD..].iter().all(|x| *x == 0.),
                "padding became trainable in {name}"
            );
        }
    }
    t.buffers
        .grad_clip
        .clip(&t.runtime.stream, &t.runtime.optimizer)?;
    let components = t.buffers.grad_clip.norm_components(
        &t.runtime.stream,
        &t.buffers.backward,
        &t.buffers.next_latent_grads,
        &t.buffers.muon_tables,
    )?;
    for (name, actual) in measured {
        let component = components
            .iter()
            .find(|c| c["name"].as_str() == Some(&name))
            .expect("mHC group missing from global norm");
        let reported = component["squared_l2"].as_f64().unwrap();
        assert!(
            (reported - actual).abs() <= actual * 5e-5 + 1e-15,
            "global clipping must use the actual masked gradient: {name} measured={actual} reported={reported}"
        );
    }
    for layer in 0..GPT2_N_LAYER {
        for matrix in [
            "attn_qkv.weight",
            "attn_c_proj.weight",
            "mlp_up.weight",
            "mlp_down.weight",
        ] {
            let name = format!("blocks.{layer}.{matrix}");
            let c = components
                .iter()
                .find(|c| c["name"].as_str() == Some(&name))
                .expect("missing old backbone branch");
            assert_eq!(c["finite"], true, "nonfinite {name}");
            assert!(c["l2"].as_f64().unwrap() > 0., "mHC disconnected {name}");
        }
    }
    for matrix in ["input_projection", "transition", "output_projection"] {
        let name = format!("next_latent.{matrix}.weight");
        let c = components
            .iter()
            .find(|c| c["name"].as_str() == Some(&name))
            .expect("NextLat was omitted");
        assert!(c["l2"].as_f64().unwrap() > 0. && c["finite"] == true);
    }
    // A bad final learned readout must reject the entire shared update.
    let bad = named.iter().find(|(n, _)| *n == "mhc.head.bias").unwrap().1;
    unsafe {
        cuda_core::memory::memset_d8_async(
            bad.cu_deviceptr(),
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    let before = t.buffers.optimizer_state.step;
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
    assert_eq!(before, t.buffers.optimizer_state.step);
    t.materialize_evaluation_weights()?;
    let loss = t.eval_token_losses(&b)?;
    assert_eq!(loss, t.eval_token_losses(&b)?);
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/mhc-full-roundtrip.ckpt");
    std::fs::create_dir_all(path.parent().unwrap())?;
    t.save_model(&path)?;
    t.load_model(&path)?;
    assert_eq!(t.mhc.as_ref().unwrap().named_gradients().len(), 198);
    assert_eq!(loss, t.eval_token_losses(&b)?);
    for cut in [1, 63, 64, 127, 128, 1023, 1024, 2015, 2047] {
        let mut altered = windows.clone();
        for seq in 0..GPT2_BATCH_SIZE {
            for x in
                &mut altered[seq * (GPT2_SEQ_LEN + 1) + cut + 1..(seq + 1) * (GPT2_SEQ_LEN + 1)]
            {
                *x = 28001;
            }
        }
        let b2 = t.batch_from_windows(&altered, GPT2_BATCH_SIZE)?;
        let other = t.eval_token_losses(&b2)?;
        for seq in 0..GPT2_BATCH_SIZE {
            let start = seq * GPT2_SEQ_LEN;
            assert_eq!(&loss[start..start + cut], &other[start..start + cut]);
        }
    }
    let mut changed_sequence = windows.clone();
    changed_sequence[(GPT2_BATCH_SIZE - 1) * (GPT2_SEQ_LEN + 1)..].fill(28001);
    let independent = t.batch_from_windows(&changed_sequence, GPT2_BATCH_SIZE)?;
    let independent_loss = t.eval_token_losses(&independent)?;
    assert_eq!(
        &loss[..(GPT2_BATCH_SIZE - 1) * GPT2_SEQ_LEN],
        &independent_loss[..(GPT2_BATCH_SIZE - 1) * GPT2_SEQ_LEN],
        "changing one sequence must preserve every other sequence"
    );
    println!(
        "full_mhc_model_verified boundaries=32 streams=4 hyperhead=learned groups=198 old_branches=all nextlat=active global_rejection=true averaged_replay=exact causality=true fixture_updates=2 prefix_cuts=9 cross_sequence=exact"
    );
    Ok(())
}

#[test]
#[ignore = "native full B8/L16/d2048 recomputation, all33 original-vs-rebuilt tapes, no training screen"]
fn native_mhc_checkpoint_windows_preserve_every_forward_and_backward_operand() -> AppResult {
    let mut t = Trainer::new_mhc(0x47505432)?;
    let mhc = t.mhc.as_mut().unwrap();
    assert_eq!(
        mhc.checkpoints.iter().map(|c| c.index).collect::<Vec<_>>(),
        vec![0, 2, 6, 10, 14, 18, 22, 26, 30]
    );
    assert_eq!(mhc.tapes.len(), 4);
    mhc.verify_recomputation();
    let windows = (0..GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1))
        .map(|i| {
            if i % (GPT2_SEQ_LEN + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let batch = t.batch_from_windows(&windows, GPT2_BATCH_SIZE)?;
    t.materialize_training_weights()?;
    t.forward_step(&batch)?;
    t.enqueue_numeric_probe_backward(&batch)?;
    let mhc = t.mhc.as_ref().unwrap();
    assert_eq!(
        mhc.recompute_verification.as_ref().unwrap().reconstructed,
        vec![true; 33],
        "every window, including the final HyperHead window, must be rebuilt and compared"
    );
    assert_eq!(t.buffers.optimizer_state.step, 0);
    for (name, gradient) in mhc.named_gradients() {
        let values = gradient.to_host_vec(&t.runtime.stream)?;
        assert!(values.iter().all(|x| x.is_finite()), "nonfinite {name}");
        assert!(values.iter().any(|x| *x != 0.), "disconnected {name}");
    }
    println!(
        "full_mhc_recomputation_verified checkpoints=9 workspace_tapes=4 original_and_rebuilt_tapes=33 input_quantization_inverse_raw_coefficients=bit_exact stochastic_backward_seed_order=unchanged all198_gradients=active updates=0"
    );
    Ok(())
}
