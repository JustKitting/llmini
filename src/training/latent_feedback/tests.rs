use super::*;
use crate::training::{TokenDataLoader, Trainer};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use std::path::Path;

fn batch(t: &Trainer) -> crate::AppResult<crate::training::TokenBatch> {
    let v = TokenDataLoader::fineweb_validation_tokens()?;
    let windows = v
        .chunks_exact(GPT2_SEQ_LEN + 1)
        .cycle()
        .take(GPT2_BATCH_SIZE)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    t.batch_from_windows(&windows, GPT2_BATCH_SIZE)
}

fn state_bits(
    f: &Feedback,
    r: &crate::training::runtime::Runtime,
) -> crate::AppResult<Vec<Vec<u32>>> {
    let s = r.stream.as_ref();
    let mut out = Vec::new();
    for b in [
        &f.state.z_master,
        &f.state.x_master,
        &f.state.second_momentum,
    ] {
        out.push(b.to_host_vec(s)?.into_iter().map(f32::to_bits).collect());
    }
    for b in [&f.state.momentum, &f.state.variance] {
        out.push(b.to_host_vec(s)?.into_iter().map(u32::from).collect());
    }
    for c in [
        &f.state.symexp_lin.exponential,
        &f.state.symexp_lin.linear,
        &f.state.symexp_lin.curvature,
    ] {
        for b in [&c.z_master, &c.x_master, &c.first, &c.second] {
            out.push(b.to_host_vec(s)?.into_iter().map(f32::to_bits).collect());
        }
    }
    Ok(out)
}

#[test]
#[ignore = "full approximately1B native CUDA feedback contract; fixture updates are not convergence evidence"]
fn native_latent_feedback_recurrence_gradients_conditional_update_checkpoint() -> crate::AppResult {
    let mut t = Trainer::new_feedback(0x47505432)?;
    println!("feedback_test_stage=initialized");
    assert!(t.engram.is_none());
    let b = batch(&t)?;
    let one = t.eval_token_losses_for_passes(&b, 1)?;
    println!("feedback_test_stage=one_pass");
    let two = t.eval_token_losses_for_passes(&b, 2)?;
    println!("feedback_test_stage=two_pass");
    let four = t.eval_token_losses_for_passes(&b, 4)?;
    println!("feedback_test_stage=four_pass");
    assert!(one.iter().chain(&two).chain(&four).all(|v| v.is_finite()));
    for seq in 0..GPT2_BATCH_SIZE {
        let i = seq * GPT2_SEQ_LEN;
        assert_eq!(
            one[i].to_bits(),
            two[i].to_bits(),
            "first position must receive no feedback"
        );
        assert_eq!(one[i].to_bits(), four[i].to_bits());
    }
    assert!(
        one.iter()
            .zip(&two)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
    assert!(
        two.iter()
            .zip(&four)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
    let eight = t.eval_token_losses_for_passes(&b, 8)?;
    println!("feedback_test_stage=eight_pass");
    for seq in 0..GPT2_BATCH_SIZE {
        let i = seq * GPT2_SEQ_LEN;
        assert_eq!(
            &four[i..i + 4],
            &eight[i..i + 4],
            "first P positions must converge after P passes"
        );
    }
    let v = TokenDataLoader::fineweb_validation_tokens()?;
    let original = v
        .chunks_exact(GPT2_SEQ_LEN + 1)
        .cycle()
        .take(GPT2_BATCH_SIZE)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    for cut in [1, 63, 64, 127, 128, 1023, 1024, 2015, 2047] {
        let mut altered = original.clone();
        for seq in 0..GPT2_BATCH_SIZE {
            let start = seq * (GPT2_SEQ_LEN + 1);
            for token in &mut altered[start + cut + 1..start + GPT2_SEQ_LEN + 1] {
                *token = if *token == 123 { 124 } else { 123 };
            }
        }
        let changed_batch = t.batch_from_windows(&altered, GPT2_BATCH_SIZE)?;
        let changed = t.eval_token_losses_for_passes(&changed_batch, 4)?;
        for seq in 0..GPT2_BATCH_SIZE {
            let i = seq * GPT2_SEQ_LEN;
            assert_eq!(
                &four[i..i + cut],
                &changed[i..i + cut],
                "future prefix cut={cut}"
            );
        }
    }
    let mut altered_sequence = original.clone();
    for token in &mut altered_sequence[GPT2_SEQ_LEN + 1..2 * (GPT2_SEQ_LEN + 1)] {
        *token = if *token == 123 { 124 } else { 123 };
    }
    let changed_batch = t.batch_from_windows(&altered_sequence, GPT2_BATCH_SIZE)?;
    let changed = t.eval_token_losses_for_passes(&changed_batch, 4)?;
    assert_eq!(
        &four[..GPT2_SEQ_LEN],
        &changed[..GPT2_SEQ_LEN],
        "sequence independence"
    );
    t.forward_for_passes(&b, 2)?;
    super::oracle_tests::assert_projection(&t)?;
    println!("feedback_test_stage=oracle");
    t.enqueue_numeric_probe_backward(&b)?;
    println!("feedback_test_stage=backward");
    let norm = t
        .buffers
        .grad_clip
        .clip(&t.runtime.stream, &t.runtime.optimizer)?;
    assert!(norm.norm.is_finite());
    let components = t.buffers.grad_clip.norm_components(
        &t.runtime.stream,
        &t.buffers.backward,
        &t.buffers.next_latent_grads,
        &t.buffers.muon_tables,
    )?;
    let mut required = Vec::new();
    for layer in 0..16 {
        for suffix in [
            "attn_qkv.weight",
            "attn_c_proj.weight",
            "mlp_up.weight",
            "mlp_down.weight",
        ] {
            required.push(format!("blocks.{layer}.{suffix}"));
        }
    }
    for suffix in ["input_projection", "transition", "output_projection"] {
        required.push(format!("next_latent.{suffix}.weight"));
    }
    assert_eq!(required.len(), 67);
    for name in required {
        let component = components.iter().find(|c| c["name"] == name).unwrap();
        assert!(
            component["finite"] == true && component["l2"].as_f64().unwrap() > 0.,
            "inactive {name}"
        );
    }
    let f = t.latent_feedback.as_ref().unwrap();
    let g = f.grad.to_host_vec(t.runtime.stream.as_ref())?;
    assert!(g.iter().all(|v| v.is_finite()) && g.iter().any(|v| *v != 0.0));
    assert!(
        t.buffers
            .muon_tables
            .matrix_grad_ranges()
            .iter()
            .any(|&(p, n)| p == f.grad.cu_deviceptr() && n as usize == GPT2_N_EMBD * GPT2_N_EMBD)
    );
    for control in [
        &f.state.symexp_lin.exponential,
        &f.state.symexp_lin.linear,
        &f.state.symexp_lin.curvature,
    ] {
        let g = control.grad.to_host_vec(t.runtime.stream.as_ref())?[0];
        assert!(
            g.is_finite() && g != 0.0,
            "active fusion control gradient {g}"
        );
    }
    let active = t.train_step_with_passes(&b, true, 2)?;
    println!("feedback_test_stage=active_update");
    assert!(!active.optimizer.update_skipped);
    let before = state_bits(t.latent_feedback.as_ref().unwrap(), &t.runtime)?;
    let inactive = t.train_step_with_passes(&b, true, 1)?;
    println!("feedback_test_stage=inactive_update");
    assert!(!inactive.optimizer.update_skipped);
    assert_eq!(t.buffers.optimizer_state.step, 2);
    assert_eq!(
        before,
        state_bits(t.latent_feedback.as_ref().unwrap(), &t.runtime)?,
        "single-pass updates must preserve every fusion master/momentum/control state"
    );
    assert!(t.completed_gradient_components(2, inactive.optimizer.grad_norm)?["passed"] == true);
    t.materialize_evaluation_weights()?;
    let expected = t.eval_token_losses_for_passes(&b, 4)?;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/latent-feedback-native-roundtrip.ckpt");
    t.save_model(&path)?;
    t.load_model(&path)?;
    println!("feedback_test_stage=checkpoint_reload");
    assert!(t.latent_feedback.is_some() && t.engram.is_none());
    assert_eq!(
        t.buffers.optimizer_state.step, 0,
        "model-only reload must not pretend to continue"
    );
    assert_eq!(expected, t.eval_token_losses_for_passes(&b, 4)?);
    let f = t.latent_feedback.as_ref().unwrap();
    let nan = [f32::NAN];
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            f.state.z_master.cu_deviceptr(),
            nan.as_ptr(),
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    t.runtime.stream.synchronize()?;
    let rejected = t.train_step_with_passes(&b, true, 2)?;
    assert!(rejected.optimizer.update_skipped && rejected.optimizer.skip_non_finite);
    assert_eq!(
        t.buffers.optimizer_state.step, 0,
        "nonfinite fusion must reject the entire update"
    );
    println!(
        "native_feedback_contract_passed full_model=true checkpoint_replay=true inactive_state_exact=true base_matrix_groups=64 nextlat_matrix_groups=3 prefix_cuts=9 sequence_independence=true"
    );
    Ok(())
}

#[test]
fn feedback_pass_sampler_is_deterministic_and_matches_reference_mix() {
    let count = (0..100000)
        .filter(|&step| training_passes(0x47505432, step) == 1)
        .count();
    assert!((14500..15500).contains(&count), "one-pass count {count}");
    for step in 0..1000 {
        assert_eq!(training_passes(7, step), training_passes(7, step));
    }
}
