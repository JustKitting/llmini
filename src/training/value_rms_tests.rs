//! Complete value-RMS graph contract, declared before production integration.
#[path = "value_rms_tests/attention.rs"]
mod attention;
use super::{
    Trainer,
    optimizer_apply::{WeightUpdateArgs, apply_weight_updates},
};
use crate::AppResult;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use sha2::{Digest, Sha256};
use std::cell::RefCell;

pub(in crate::training) struct ValueAudit {
    expected: RefCell<Vec<Option<[u8; 32]>>>,
    checked: RefCell<Vec<bool>>,
    point_checked: RefCell<Vec<bool>>,
    point_module: rust_kernels_cuda::value_rms::ValueRmsModule,
    point_flags: RefCell<DeviceBuffer<u32>>,
}
impl gpt2_nvfp4::ValueProjectionObserver for ValueAudit {
    fn observe(
        &self,
        block: usize,
        s: &CudaStream,
        qkv: &DeviceBuffer<f32>,
        n: usize,
        d: usize,
        stride: usize,
        recomputed: bool,
    ) -> Result<(), DriverError> {
        let all = qkv.to_host_vec(s)?;
        let mut hash = Sha256::new();
        for row in 0..n {
            let value = &all[row * stride + 2 * d..row * stride + 3 * d];
            assert!(value.iter().all(|x| x.is_finite()));
            hash.update(unsafe { std::slice::from_raw_parts(value.as_ptr().cast::<u8>(), d * 4) });
        }
        let digest: [u8; 32] = hash.finalize().into();
        if recomputed {
            assert_eq!(
                self.expected.borrow()[block],
                Some(digest),
                "mixed-V native recomputation layer{block}"
            );
            self.checked.borrow_mut()[block] = true;
        } else {
            self.expected.borrow_mut()[block] = Some(digest);
            self.checked.borrow_mut()[block] = false;
            self.point_checked.borrow_mut()[block] = false;
        }
        Ok(())
    }

    fn represented_point(
        &self,
        block: usize,
        s: &CudaStream,
        raw: &DeviceBuffer<f32>,
        stored: &DeviceBuffer<u16>,
        n: usize,
        d: usize,
        stride: usize,
        kda: bool,
    ) -> Result<(), DriverError> {
        assert!(
            self.checked.borrow()[block],
            "raw replay must be verified first"
        );
        let mut flags = self.point_flags.borrow_mut();
        self.point_module.check_stored_point(
            s,
            raw,
            stored,
            &mut flags,
            n as u32,
            d as u32,
            stride as u32,
            kda,
        )?;
        let flags = flags.to_host_vec(s)?;
        assert!(
            flags[..n * (d / 64)].iter().all(|&v| v == 0),
            "reconstructed represented normalized V differs from the actual forward tape in block{block}"
        );
        self.point_checked.borrow_mut()[block] = true;
        Ok(())
    }
}
pub(in crate::training) fn hash(
    s: &CudaStream,
    b: &DeviceBuffer<f32>,
) -> Result<[u8; 32], DriverError> {
    let values = b.to_host_vec(s)?;
    Ok(Sha256::digest(unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
    })
    .into())
}

impl Trainer {
    pub(in crate::training) fn enable_value_projection_verification(&mut self) -> AppResult {
        self.value_projection_audit = Some(ValueAudit {
            expected: RefCell::new(vec![None; 16]),
            checked: RefCell::new(vec![false; 16]),
            point_checked: RefCell::new(vec![false; 16]),
            point_module: self.runtime.value_rms_point_check_module()?,
            point_flags: RefCell::new(DeviceBuffer::zeroed(
                &self.runtime.stream,
                GPT2_BATCH_SIZE * GPT2_SEQ_LEN * gpt2_nvfp4::GPT2_N_HEAD,
            )?),
        });
        Ok(())
    }

    pub(in crate::training) fn value_projection_recomputation_verified(
        &self,
    ) -> (Vec<bool>, Vec<bool>) {
        let audit = self
            .value_projection_audit
            .as_ref()
            .expect("value projection verification must be enabled");
        (
            audit.checked.borrow().clone(),
            audit.point_checked.borrow().clone(),
        )
    }
}

#[test]
#[ignore = "actual fullB8/L16/d2048 value-RMS, full/KDA/XSA/fixed-share/common optimizer/checkpoint/causality"]
fn native_value_rms_complete_graph_recomputation_cache_checkpoint_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_VALUE_RMS);
    assert!(!gpt2_nvfp4::GPT2_BOUNDED_SWIGLU);
    let mut t = Trainer::new_value_rms(0x47505432)?;
    assert!(t.engram.is_none() && t.mhc.is_none() && t.latent_feedback.is_none());
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
        let step = t.train_step(&b, true)?;
        assert!(
            step.finite && !step.optimizer.update_skipped && step.optimizer.grad_norm.is_finite()
        );
    }
    t.materialize_training_weights()?;
    t.value_projection_audit = Some(ValueAudit {
        expected: RefCell::new(vec![None; 16]),
        checked: RefCell::new(vec![false; 16]),
        point_checked: RefCell::new(vec![false; 16]),
        point_module: t.runtime.value_rms_point_check_module()?,
        point_flags: RefCell::new(DeviceBuffer::zeroed(
            &t.runtime.stream,
            GPT2_BATCH_SIZE * GPT2_SEQ_LEN * gpt2_nvfp4::GPT2_N_HEAD,
        )?),
    });
    t.forward_step(&b)?;
    let raw_before = hash(&t.runtime.stream, &t.buffers.value_residual)?;
    t.enqueue_numeric_probe_backward(&b)?;
    assert_eq!(
        t.value_projection_audit
            .as_ref()
            .unwrap()
            .checked
            .borrow()
            .as_slice(),
        &[true; 16]
    );
    assert!(
        t.value_projection_audit
            .as_ref()
            .unwrap()
            .point_checked
            .borrow()
            .iter()
            .all(|&v| v),
        "all16 actual normalized forward tape points must match the reverse reconstruction"
    );
    assert_eq!(
        raw_before,
        hash(&t.runtime.stream, &t.buffers.value_residual)?,
        "raw first-value cache was overwritten by a gradient"
    );
    t.value_projection_audit = None;
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
            bad.cu_deviceptr() + (2 * GPT2_N_EMBD * 4) as u64,
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
        .join("target/checks/20261116/value-rms-represented-component-roundtrip.ckpt");
    t.save_model(&path)?;
    t.load_model(&path)?;
    assert_eq!(
        before,
        t.eval_token_losses(&b)?,
        "averaged native checkpoint replay"
    );
    let legacy = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/runs/20261114_080934Z_fineweb_21600s/model.ckpt");
    assert!(
        t.load_model(&legacy).is_err(),
        "unnormalized checkpoint was silently reinterpreted"
    );
    assert_eq!(before, t.eval_token_losses(&b)?);
    let old_point = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/checks/20261115/value-rms-full-roundtrip.ckpt");
    let error = t.load_model(&old_point).unwrap_err().to_string();
    assert!(
        error.contains("checkpoint value-normalization graph does not match"),
        "old value-RMS point must reject for its numerical descriptor: {error}"
    );
    assert_eq!(
        before,
        t.eval_token_losses(&b)?,
        "rejected FP32 reverse-point checkpoint mutated the represented component"
    );
    println!(
        "historical_value_RMS_reverse_policy_rejected component_graph=unchanged descriptor_checked_before_upload=true"
    );
    for cut in [1, 31, 63, 64, 127, 128, 1023, 2015, 2047] {
        let mut changed = flat.clone();
        for window in changed.chunks_exact_mut(GPT2_SEQ_LEN + 1) {
            window[cut + 1..].fill(28001);
        }
        let changed = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
        let loss = t.eval_token_losses(&changed)?;
        for seq in 0..GPT2_BATCH_SIZE {
            let start = seq * GPT2_SEQ_LEN;
            assert_eq!(&before[start..start + cut], &loss[start..start + cut]);
        }
    }
    let mut changed = flat.clone();
    changed[..GPT2_SEQ_LEN + 1].fill(28002);
    let changed = t.batch_from_windows(&changed, GPT2_BATCH_SIZE)?;
    assert_eq!(
        &before[GPT2_SEQ_LEN..],
        &t.eval_token_losses(&changed)?[GPT2_SEQ_LEN..]
    );
    println!(
        "complete_value_rms_model_verified layers=16 full_KDA_XSA_fixed_share=true native_mixed_V_recomputation=bit_exact represented_normalized_points16=bit_exact raw_first_cache=unchanged all_old_branches_NextLat=active global_rejection=true averaged_checkpoint=bit_exact legacy_graph_rejected=true causality_cases=9 cross_sequence=bit_exact fixture_updates=2"
    );
    Ok(())
}
