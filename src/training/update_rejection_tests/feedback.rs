//! Full real-train_step coverage declared in latent_feedback_repaired_native_contract_20261115.md.
use super::{AppResult, Audit, B, BadGradient, D, S, Snapshot, Trainer, json};

fn word(t: &Trainer, ptr: u64) -> AppResult<f32> {
    let mut value = [0f32];
    unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            value.as_mut_ptr(),
            ptr,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    t.runtime.stream.synchronize()?;
    Ok(value[0])
}
fn set_word(t: &Trainer, ptr: u64, value: f32) -> AppResult {
    let values = [value];
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            ptr,
            values.as_ptr(),
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    t.runtime.stream.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "complete B8/L16/d2048 feedback actual P2 train_step; Ember/all-state faults, P1 inactivity and recovery"]
fn native_standalone_feedback_global_rejection_preserves_all_training_state() -> AppResult {
    assert_eq!(
        (B, S, D, gpt2_nvfp4::GPT2_N_LAYER, gpt2_nvfp4::GPT2_N_HEAD),
        (8, 2048, 2048, 16, 32)
    );
    let mut t = Trainer::new_feedback(0x47505432)?;
    assert!(t.engram.is_none() && t.mhc.is_none());
    assert!(matches!(
        &t.buffers.optimizer_state.token_embedding,
        crate::training::optimizer_state::TokenEmbeddingState::Ember(_)
    ));
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
        let stats = t.train_step_with_passes(&b, true, 2)?;
        assert!(
            stats.finite
                && !stats.optimizer.update_skipped
                && stats.optimizer.grad_norm.is_finite()
        );
    }
    assert_eq!(t.buffers.optimizer_state.step, 2);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/feedback-state-integrity");
    std::fs::create_dir_all(&out)?;
    let mut results = Vec::new();
    // None uses the actual materialization/GEMM producer to propagate a bad master.
    // It never mutates a matrix gradient behind a valid cached GEMM norm.
    for (name, case) in [
        ("fusion_master", None),
        ("fusion_exponential", Some(BadGradient::FeedbackExponential)),
        ("fusion_curvature", Some(BadGradient::FeedbackCurvature)),
        ("late_backbone_bias", Some(BadGradient::FeedbackLateBias)),
    ] {
        let master = t
            .latent_feedback
            .as_ref()
            .unwrap()
            .state
            .z_master
            .cu_deviceptr();
        let original = if case.is_none() {
            let original = word(&t, master)?;
            assert!(original.is_finite());
            set_word(&t, master, f32::NAN)?;
            Some(original)
        } else {
            None
        };
        t.update_rejection_audit = Some(Audit {
            case,
            expected: None,
        });
        let stats = t.train_step_with_passes(&b, true, 2)?;
        let expected = t.update_rejection_audit.take().unwrap().expected.unwrap();
        let actual = Snapshot::capture(&t)?;
        let before = expected.json();
        let after = actual.json();
        std::fs::write(
            out.join(format!("{name}-before.json")),
            serde_json::to_vec_pretty(&before)?,
        )?;
        std::fs::write(
            out.join(format!("{name}-after.json")),
            serde_json::to_vec_pretty(&after)?,
        )?;
        assert!(
            stats.optimizer.update_skipped && stats.optimizer.skip_non_finite,
            "{name}"
        );
        if case.is_some() {
            assert!(stats.finite, "vector fault must follow a finite forward");
        }
        assert_eq!(
            expected, actual,
            "persistent update state changed during {name}"
        );
        assert_eq!(t.buffers.optimizer_state.step, 2);
        assert_eq!(before["device_buffers"], 2814);
        results.push(json!({"case":name,"passed":true,"loss_finite":stats.finite,
            "before":before["root_sha256"],"after":after["root_sha256"],
            "buffers":before["device_buffers"],"counts":before["counts"]}));
        if let Some(value) = original {
            set_word(&t, master, value)?;
        }
        println!(
            "complete_feedback_state_rejection case={name} buffers=2814 state=bit_exact applied_updates=2"
        );
    }
    let recovery = t.train_step_with_passes(&b, true, 2)?;
    assert!(
        recovery.finite
            && !recovery.optimizer.update_skipped
            && recovery.optimizer.grad_norm.is_finite()
    );
    assert_eq!(t.buffers.optimizer_state.step, 3);

    t.update_rejection_audit = Some(Audit {
        case: None,
        expected: None,
    });
    let inactive = t.train_step_with_passes(&b, true, 1)?;
    assert!(inactive.finite && !inactive.optimizer.update_skipped);
    assert_eq!(t.buffers.optimizer_state.step, 4);
    let before = t.update_rejection_audit.take().unwrap().expected.unwrap();
    let after = Snapshot::capture(&t)?;
    assert_eq!(
        before.assert_parameter_unchanged(&after, "latent_feedback.fusion.weight"),
        21
    );
    let before_json = before.json();
    let after_json = after.json();
    assert_eq!(
        before_json["metadata"]["latent_feedback"]["rng"],
        after_json["metadata"]["latent_feedback"]["rng"]
    );
    std::fs::write(
        out.join("inactive-before.json"),
        serde_json::to_vec_pretty(&before_json)?,
    )?;
    std::fs::write(
        out.join("inactive-after.json"),
        serde_json::to_vec_pretty(&after_json)?,
    )?;
    let result = json!({"passed":true,"graph":"standalone_latent_feedback","embedding_optimizer":"Ember",
        "shape":{"batch":B,"context":S,"layers":16,"width":D,"heads":32},"cases":results,
        "injected_rejections":4,"clean_recovery":true,"recovery_applied_step":3,"fixture_applied_updates":4,
        "inactive_fusion_buffers_unchanged":21,"inactive_backbone_advanced":true,"inactive_fusion_rng_unchanged":true,
        "dense_forward_backward":"NVFP4","CPU_training":false,"loss_evidence":false,
        "host_fingerprint_workspace_bytes":super::fingerprint::CHUNK_BYTES});
    std::fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    println!(
        "complete_standalone_feedback_all_training_state_verified cases=4 clean_recovery=true applied_updates=4 inactive_fusion_buffers=21 embedding_optimizer=Ember loss_evidence=false"
    );
    Ok(())
}
