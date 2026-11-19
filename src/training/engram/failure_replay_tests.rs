//! Exact saved native counterexample, not a new training screen.
use crate::{AppResult, training::Trainer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use std::path::Path;

#[test]
#[ignore = "native CUDA inverse initialization contract, zero optimizer updates"]
fn native_engram_frozen_weight_inverse_reconstructs_large_parameters() -> AppResult {
    let r = crate::training::runtime::Runtime::new()?;
    let beta = crate::training::symexp_lin::beta();
    let targets = [
        0., 0.02, -0.02, 1., -1., 7.2, -7.2, 8., -8., 100., -100., 1e10,
    ];
    let mut inverse = cuda_core::DeviceBuffer::from_host(&r.stream, &targets)?;
    r.optimizer
        .symexp_lin_congruent_inverse_in_place(&r.stream, &mut inverse, beta)?;
    for (target, raw) in targets.into_iter().zip(inverse.to_host_vec(&r.stream)?) {
        assert!(raw.is_finite(), "inverse of {target} is {raw}");
        let x = f64::from(raw);
        let b = f64::from(beta);
        let reconstructed = x.signum() * ((b * x.abs()).exp() - 1. + x.abs()) / b;
        assert!(
            (reconstructed - f64::from(target)).abs() < 2e-5 * f64::from(target).abs().max(1.),
            "inverse did not converge: target={target}, raw={raw}, reconstructed={reconstructed}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "native B8 CUDA replay of the actual attempt13559 failure, zero optimizer updates"]
fn native_engram_failure_13559_exact_replay() -> AppResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checkpoint =
        root.join("target/runs/20261114_184213Z_fineweb_21600s/numerical_failure_weights.ckpt");
    crate::reference_validation::require_hash(
        &checkpoint,
        "d3f0592425c22a70fc95db1aeb9ab85278409bc001b6be2141848ec82cc0279c",
    )?;
    assert_eq!((GPT2_BATCH_SIZE, GPT2_SEQ_LEN), (8, 2048));
    let input_path = root.join("target/checks/20261115/engram-failure-13559-input.u16");
    crate::reference_validation::require_hash(
        &input_path,
        "f39dd7f801bfde4a4068d75285762173d37f8819b97906bd566aa97f2855593d",
    )?;
    let raw = std::fs::read(input_path)?;
    assert_eq!(raw.len(), (GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1) * 2);
    let contiguous = raw
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect::<Vec<_>>();
    let mut windows = Vec::new();
    for sequence in 0..GPT2_BATCH_SIZE {
        let at = sequence * GPT2_SEQ_LEN;
        windows.extend_from_slice(&contiguous[at..at + GPT2_SEQ_LEN + 1]);
    }
    let mut t = Trainer::new(0x47505432)?;
    t.load_model(&checkpoint)?;
    let b = t.batch_from_windows(&windows, GPT2_BATCH_SIZE)?;
    t.forward_step(&b)?;
    let stream = t.runtime.stream.as_ref();
    let e = &t.engram.as_ref().unwrap().modules[0];
    let mut counts = Vec::new();
    for (name, buffer) in [
        ("input", &e.hidden),
        ("key", &e.key),
        ("value", &e.value),
        ("gated", &e.gated),
    ] {
        let data = buffer.to_host_vec(stream)?;
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        let first = data.iter().position(|v| !v.is_finite());
        println!("failure_replay {name} nonfinite={bad} first={first:?}");
        counts.push((name, bad));
    }
    assert_eq!(counts[0].1, 0, "upstream block must be finite");
    assert_eq!(counts[1].1, 0, "Engram key must be finite");
    assert_eq!(
        counts[2].1, 0,
        "actual saved Engram value projection must be finite"
    );
    assert_eq!(
        counts[3].1, 0,
        "actual saved Engram gated value must be finite"
    );
    let losses = t.eval_token_losses(&b)?;
    assert!(losses.iter().all(|v| v.is_finite()));
    // Full native backward on the frozen model; no optimizer continuation.
    t.enqueue_numeric_probe_backward(&b)?;
    let norm = t
        .buffers
        .grad_clip
        .clip(t.runtime.stream.as_ref(), &t.runtime.optimizer)?
        .norm;
    if !norm.is_finite() {
        t.write_numerical_failure_dump(
            &b,
            &root.join("target/checks/20261115/engram-failure-replay-backward-state.tsv"),
        )?;
    }
    assert!(
        norm.is_finite() && norm > 0.0,
        "exact replay gradient norm {norm}"
    );
    assert_eq!(t.buffers.optimizer_state.step, 0);
    println!(
        "{}",
        serde_json::json!({"passed":true,"case":"actual attempt13559", "optimizer_updates":0,
        "checkpoint_sha256":"d3f0592425c22a70fc95db1aeb9ab85278409bc001b6be2141848ec82cc0279c",
        "source":"fineweb_mistral_v01_train_000011.bin","token_offset":22249472,"batch":8,"context":2048,
        "mean_primary_ce":losses.iter().map(|v| f64::from(*v)).sum::<f64>() / losses.len() as f64,
        "full_native_backward_gradient_norm":norm,"optimizer_state_replayed":false})
    );
    Ok(())
}

#[test]
#[ignore = "native Engram affine-bias storage contract, zero optimizer updates"]
fn native_engram_biases_have_vector_scale_storage() -> AppResult {
    let r = crate::training::runtime::Runtime::new()?;
    let w = super::ModuleWeights::new(&r)?;
    for index in [2, 4] {
        let tensor = &w.nv[index];
        assert_eq!(
            tensor.scale_rows, 1,
            "Engram bias {index} must use vector scales for the native affine kernel"
        );
        assert_eq!(tensor.scale_k as usize, gpt2_nvfp4::GPT2_N_EMBD);
        let scales = tensor.scales.to_host_vec(&r.stream)?;
        assert_eq!(scales[8], 255);
    }
    Ok(())
}
