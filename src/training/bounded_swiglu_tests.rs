//! Complete native model acceptance declared before bounded-SwiGLU integration.
use crate::training::optimizer_apply::{WeightUpdateArgs, apply_weight_updates};
use crate::{AppResult, training::Trainer};
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
mod initialization;
mod mlp;
pub(in crate::training) use initialization::verify_source_normal_upload;

pub(in crate::training) struct RecomputeAudit {
    expected: RefCell<Vec<Option<[u8; 32]>>>,
    checked: RefCell<Vec<bool>>,
}
impl gpt2_nvfp4::MlpProjectionObserver for RecomputeAudit {
    fn observe(
        &self,
        block: usize,
        stream: &CudaStream,
        paired: &DeviceBuffer<f32>,
        elements: usize,
        recomputed: bool,
    ) -> Result<(), DriverError> {
        let values = paired.to_host_vec(stream)?;
        assert!(values.len() >= elements && values[..elements].iter().all(|x| x.is_finite()));
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), elements * 4) };
        let hash: [u8; 32] = Sha256::digest(bytes).into();
        if recomputed {
            assert_eq!(
                self.expected.borrow()[block],
                Some(hash),
                "paired projection recomputation at layer{block}"
            );
            self.checked.borrow_mut()[block] = true;
        } else {
            self.expected.borrow_mut()[block] = Some(hash);
            self.checked.borrow_mut()[block] = false;
        }
        Ok(())
    }
}
impl Trainer {
    pub(in crate::training) fn enable_bounded_projection_verification(&mut self) {
        assert!(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU);
        self.bounded_projection_audit = Some(RecomputeAudit {
            expected: RefCell::new(vec![None; 16]),
            checked: RefCell::new(vec![false; 16]),
        });
    }
    pub(in crate::training) fn bounded_projection_recomputation_verified(&self) -> Vec<bool> {
        self.bounded_projection_audit
            .as_ref()
            .unwrap()
            .checked
            .borrow()
            .clone()
    }
}

#[test]
fn compiled_mlp_geometry_keeps_legacy_or_complete_paired_layout() {
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        assert_eq!(
            (gpt2_nvfp4::GPT2_MLP, gpt2_nvfp4::GPT2_MLP_UP),
            (5504, 11008)
        );
        assert_eq!(
            (
                gpt2_nvfp4::GPT2_MLP_ROUTE_FEATURE_TILES,
                gpt2_nvfp4::GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES
            ),
            (43, 32)
        );
    } else {
        assert_eq!(
            (gpt2_nvfp4::GPT2_MLP, gpt2_nvfp4::GPT2_MLP_UP),
            (8192, 8192)
        );
        assert_eq!(
            (
                gpt2_nvfp4::GPT2_MLP_ROUTE_FEATURE_TILES,
                gpt2_nvfp4::GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES
            ),
            (64, 48)
        );
    }
}

#[test]
#[ignore = "actual full B8/L16/d2048 paired5504 MLP, all model branches, optimizer/checkpoint/causality"]
fn native_bounded_swiglu_complete_model_gradients_recomputation_checkpoint_causality() -> AppResult
{
    assert!(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU);
    let mut t = Trainer::new_bounded_swiglu(0x47505432)?;
    assert!(t.engram.is_none() && t.mhc.is_none() && t.latent_feedback.is_none());
    for w in &t.uploaded.blocks {
        assert_eq!(w.mlp_up.weight.len, GPT2_N_EMBD * 11008);
        assert_eq!(w.mlp_up.bias.len, 11008);
        assert_eq!(w.mlp_down.weight.len, GPT2_N_EMBD * 5504);
    }
    let flat = (0..GPT2_BATCH_SIZE * (GPT2_SEQ_LEN + 1))
        .map(|i| {
            if i % (GPT2_SEQ_LEN + 1) == 0 {
                1
            } else {
                (32 + i * 37 % 28000) as u16
            }
        })
        .collect::<Vec<_>>();
    let b = t.batch_from_windows(&flat, GPT2_BATCH_SIZE)?;
    for _ in 0..2 {
        let result = t.train_step(&b, true)?;
        assert!(result.finite && result.optimizer.grad_norm.is_finite());
        assert!(!result.optimizer.update_skipped);
    }
    t.materialize_training_weights()?;
    t.enable_bounded_projection_verification();
    t.forward_step(&b)?;
    t.enqueue_numeric_probe_backward(&b)?;
    assert_eq!(
        t.bounded_projection_recomputation_verified(),
        vec![true; 16]
    );
    for (layer, g) in t.buffers.backward.blocks.iter().enumerate() {
        let paired = g.d_mlp_c_fc_weight.to_host_vec(&t.runtime.stream)?;
        assert_eq!(paired.len(), GPT2_N_EMBD * 11008);
        for (half, values) in paired.chunks_exact(GPT2_N_EMBD * 5504).enumerate() {
            assert!(
                values.iter().all(|x| x.is_finite()),
                "nonfinite layer{layer} half{half}"
            );
            assert!(
                values.iter().any(|x| *x != 0.),
                "dead layer{layer} half{half}"
            );
        }
        for g in [
            &g.d_mlp_c_fc_bias,
            &g.d_mlp_c_proj_weight,
            &g.d_mlp_c_proj_bias,
        ] {
            let v = g.to_host_vec(&t.runtime.stream)?;
            assert!(v.iter().all(|x| x.is_finite()));
            assert!(v.iter().any(|x| *x != 0.));
        }
    }
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
    // A bad element in the late UP half must be inside the common guard;
    // checking only the first5504 bias entries would miss this failure.
    let bad = &t.buffers.backward.blocks[15].d_mlp_c_fc_bias;
    unsafe {
        cuda_core::memory::memset_d8_async(
            bad.cu_deviceptr() + (5504 * 4) as u64,
            255,
            4,
            t.runtime.stream.cu_stream(),
        )?;
    }
    let step_before = t.buffers.optimizer_state.step;
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
    assert_eq!(step_before, t.buffers.optimizer_state.step);
    t.materialize_evaluation_weights()?;
    let before = t.eval_token_losses(&b)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/bounded-swiglu-full-roundtrip.ckpt");
    t.save_model(&path)?;
    t.load_model(&path)?;
    let actual = t.eval_token_losses(&b)?;
    assert_eq!(before, actual, "complete averaged checkpoint replay");
    let legacy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/runs/20261114_080934Z_fineweb_21600s/model.ckpt");
    assert!(
        t.load_model(&legacy).is_err(),
        "legacy8192 ReLU weights must not be reinterpreted as paired5504"
    );
    assert_eq!(actual, t.eval_token_losses(&b)?);
    for cut in [1, 31, 63, 64, 127, 128, 1023, 2015, 2047] {
        let mut changed = flat.clone();
        for window in changed.chunks_exact_mut(GPT2_SEQ_LEN + 1) {
            window[cut + 1..].fill(28001);
        }
        let changed = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
        let losses = t.eval_token_losses(&changed)?;
        for s in 0..GPT2_BATCH_SIZE {
            let start = s * GPT2_SEQ_LEN;
            assert_eq!(&actual[start..start + cut], &losses[start..start + cut]);
        }
    }
    let mut independent = flat.clone();
    independent[..GPT2_SEQ_LEN + 1].fill(28002);
    let independent = t.batch_from_windows(&independent, GPT2_BATCH_SIZE)?;
    assert_eq!(
        &actual[GPT2_SEQ_LEN..],
        &t.eval_token_losses(&independent)?[GPT2_SEQ_LEN..]
    );
    println!(
        "complete_bounded_swiglu_model_verified layers=16 d=2048 gate_up=11008 product=5504 routing=32of43 all_gate_up_down_and_old_branches=active nextlat=active all16_recomputed_projections=bit_exact averaged_checkpoint=bit_exact legacy_shape_rejected=true causality_cases=9 cross_sequence=bit_exact fixture_updates=2"
    );
    Ok(())
}
