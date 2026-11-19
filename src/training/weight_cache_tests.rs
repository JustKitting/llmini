use cuda_core::CudaStream;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN};

use super::{TokenDataLoader, Trainer};
use crate::AppResult;
use crate::upload::{UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNvfp4};

struct Snapshot {
    name: String,
    bytes: Vec<u8>,
    scales: Vec<u8>,
    global: u32,
}

#[test]
#[ignore = "requires generated sm_120a PTX and the full approximately 1B model; no training"]
fn learned_scale_gradients_reach_the_global_update_guard() -> AppResult {
    let mut trainer = Trainer::new(0x4750_5432)?;
    let stream = trainer.runtime.stream.as_ref();
    let slot = trainer
        .buffers
        .muon_tables
        .all
        .host_symexp_lin_slots
        .iter()
        .find(|slot| slot.exponential_grad != 0)
        .unwrap();
    let (exponential, linear) = (slot.exponential_grad, slot.linear_grad);
    let write = |pointer: u64, value: f32| -> AppResult {
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                pointer,
                &value as *const f32,
                size_of::<f32>(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        Ok(())
    };
    write(exponential, 3.0)?;
    write(linear, 4.0)?;
    let result = trainer
        .buffers
        .grad_clip
        .clip(stream, &trainer.runtime.optimizer)?;
    assert!(
        (result.norm - 5.0).abs() < 1.0e-5,
        "learned controls must contribute to the global norm: {}",
        result.norm
    );
    write(exponential, f32::NAN)?;
    let result = trainer
        .buffers
        .grad_clip
        .clip(stream, &trainer.runtime.optimizer)?;
    assert!(
        !result.norm.is_finite(),
        "nonfinite learned-control gradients must reach the update guard"
    );
    let decision = trainer
        .buffers
        .optimizer_state
        .should_skip_update(Some(3.0), result.norm);
    assert!(decision.non_finite && decision.skipped);
    assert_eq!(trainer.buffers.optimizer_state.next_step(), 1);
    Ok(())
}

fn canon_snapshot(model: &UploadedModel, stream: &CudaStream) -> AppResult<Vec<Vec<u32>>> {
    model
        .blocks
        .iter()
        .flat_map(|b| [&b.canon_a, &b.canon_c])
        .map(|t| {
            Ok(t.weight
                .to_host_vec(stream)?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        })
        .collect()
}

fn tensors(model: &UploadedModel) -> Vec<(String, &UploadedNvfp4)> {
    fn norm<'a>(out: &mut Vec<(String, &'a UploadedNvfp4)>, name: &str, x: &'a UploadedLayerNorm) {
        out.push((format!("{name}.weight"), &x.weight));
        out.push((format!("{name}.bias"), &x.bias));
    }
    fn linear<'a>(out: &mut Vec<(String, &'a UploadedNvfp4)>, name: &str, x: &'a UploadedLinear) {
        out.push((format!("{name}.weight"), &x.weight));
        out.push((format!("{name}.bias"), &x.bias));
    }
    let mut out = vec![
        ("token_embedding".into(), &model.token_embedding),
        ("xsa_alphas".into(), &model.xsa_alphas),
    ];
    norm(&mut out, "ln_f", &model.ln_f);
    norm(&mut out, "next_latent.norm", &model.next_latent.norm);
    linear(
        &mut out,
        "next_latent.input",
        &model.next_latent.input_projection,
    );
    linear(
        &mut out,
        "next_latent.transition",
        &model.next_latent.transition,
    );
    linear(
        &mut out,
        "next_latent.output",
        &model.next_latent.output_projection,
    );
    for (i, b) in model.blocks.iter().enumerate() {
        norm(&mut out, &format!("blocks.{i}.ln_1"), &b.ln_1);
        norm(&mut out, &format!("blocks.{i}.ln_2"), &b.ln_2);
        linear(&mut out, &format!("blocks.{i}.attn_qkv"), &b.attn_qkv);
        linear(&mut out, &format!("blocks.{i}.attn_c_proj"), &b.attn_c_proj);
        linear(&mut out, &format!("blocks.{i}.mlp_up"), &b.mlp_up);
        linear(&mut out, &format!("blocks.{i}.mlp_down"), &b.mlp_down);
        out.push((format!("blocks.{i}.attn_qk_scale"), &b.attn_qk_scale));
    }
    out
}

fn snapshot(model: &UploadedModel, stream: &CudaStream) -> AppResult<Vec<Snapshot>> {
    tensors(model)
        .into_iter()
        .map(|(name, t)| {
            Ok(Snapshot {
                name,
                bytes: t.bytes.to_host_vec(stream)?,
                scales: t.scales.to_host_vec(stream)?,
                global: t.global_scale.to_host_vec(stream)?[0].to_bits(),
            })
        })
        .collect()
}

fn assert_weights(
    expected: &[Snapshot],
    model: &UploadedModel,
    stream: &CudaStream,
    phase: &str,
) -> AppResult {
    let mut changed = Vec::new();
    for (saved, (name, t)) in expected.iter().zip(tensors(model)) {
        assert_eq!(saved.name, name);
        let bytes = t.bytes.to_host_vec(stream)?;
        let scales = t.scales.to_host_vec(stream)?;
        let global = t.global_scale.to_host_vec(stream)?[0].to_bits();
        if saved.bytes != bytes || saved.scales != scales || saved.global != global {
            let byte_diffs: Vec<_> = saved
                .bytes
                .iter()
                .zip(&bytes)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .collect();
            let scale_diffs: Vec<_> = saved
                .scales
                .iter()
                .zip(&scales)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .take(8)
                .collect();
            println!(
                "{phase}: {name}: payload {byte_diffs:?}, scales {scale_diffs:?}, global {:08x}->{global:08x}",
                saved.global
            );
            changed.push(name);
        }
    }
    assert!(
        changed.is_empty(),
        "{phase}: {} quantized tensors changed; first tensors: {:?}",
        changed.len(),
        &changed[..changed.len().min(12)]
    );
    println!(
        "{phase}: exact payload, scales and global scale preserved for {} quantized tensors",
        expected.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires the full 1B FineWeb model and generated sm_120a PTX"]
fn skipped_update_preserves_training_weights_and_evaluation_restore() -> AppResult {
    assert_eq!(
        (
            GPT2_BATCH_SIZE,
            GPT2_SEQ_LEN,
            GPT2_N_LAYER,
            GPT2_N_EMBD,
            GPT2_N_HEAD
        ),
        (4, 2048, 16, 2048, 32)
    );
    let mut trainer = Trainer::new(0x47505432)?;
    let mut data = TokenDataLoader::from_fineweb()?;
    let first = data.next_batch()?;
    let batch = trainer.batch_from_windows(&first.tokens, first.batch_size)?;
    let stats = trainer.train_step(&batch, true)?;
    println!(
        "initial full-model loss={} grad_norm={}",
        stats.loss, stats.optimizer.grad_norm
    );
    assert!(
        !stats.optimizer.update_skipped,
        "initial update must create the scheduled-value cache"
    );
    trainer.materialize_training_weights()?;
    let expected = snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?;
    let expected_canon = canon_snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?;
    assert_eq!(trainer.buffers.optimizer_state.next_step(), 2);

    // Prime the existing guard with stable history, then run a real second
    // FineWeb backward pass. Its finite nonzero gradients trigger a skip.
    for _ in 0..128 {
        trainer
            .buffers
            .optimizer_state
            .update_skip
            .observe(Some(0.0), 0.0);
    }
    let second = data.next_batch()?;
    let batch = trainer.batch_from_windows(&second.tokens, second.batch_size)?;
    let stats = trainer.train_step(&batch, true)?;
    assert!(stats.optimizer.update_skipped && stats.optimizer.skip_loss_spike);
    assert!(!stats.optimizer.skip_non_finite);
    assert_eq!(trainer.buffers.optimizer_state.next_step(), 2);
    trainer.materialize_training_weights()?;
    assert_weights(
        &expected,
        &trainer.uploaded,
        trainer.runtime.stream.as_ref(),
        "after skipped update",
    )?;
    assert_eq!(
        expected_canon,
        canon_snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?
    );
    trainer.materialize_evaluation_weights()?;
    trainer.materialize_training_weights()?;
    assert_weights(
        &expected,
        &trainer.uploaded,
        trainer.runtime.stream.as_ref(),
        "after evaluation restore",
    )?;
    assert_eq!(
        expected_canon,
        canon_snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?
    );

    // A subsequent real update must create a fresh cache and invalidate the
    // retained training view, even when the preceding update was skipped.
    trainer.buffers.optimizer_state.update_skip = super::update_skip::UpdateSkipState::new();
    let third = data.next_batch()?;
    let batch = trainer.batch_from_windows(&third.tokens, third.batch_size)?;
    let stats = trainer.train_step(&batch, true)?;
    assert!(!stats.optimizer.update_skipped);
    assert!(stats.finite && stats.optimizer.grad_norm.is_finite());
    assert_eq!(trainer.buffers.optimizer_state.next_step(), 3);
    trainer.materialize_training_weights()?;
    let updated = snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?;
    let updated_canon = canon_snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?;
    assert!(
        expected
            .iter()
            .zip(&updated)
            .any(|(a, b)| a.bytes != b.bytes)
    );
    trainer.materialize_evaluation_weights()?;
    trainer.materialize_training_weights()?;
    assert_weights(
        &updated,
        &trainer.uploaded,
        trainer.runtime.stream.as_ref(),
        "after normal-update evaluation restore",
    )?;
    assert_eq!(
        updated_canon,
        canon_snapshot(&trainer.uploaded, trainer.runtime.stream.as_ref())?
    );
    println!("all 32 FP32 Canon tensors preserved across skipped and normal update restoration");
    Ok(())
}

fn numeric_probe_hash(words: impl IntoIterator<Item = u32>) -> u64 {
    words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
}
fn numeric_probe_u16(
    name: &str,
    b: &cuda_core::DeviceBuffer<u16>,
    stream: &CudaStream,
) -> AppResult {
    let v = b.to_host_vec(stream)?;
    println!(
        "NUMERIC_U16 name={name} count={} hash={:016x}",
        v.len(),
        numeric_probe_hash(v.into_iter().map(u32::from))
    );
    Ok(())
}
fn numeric_probe_f32(
    name: &str,
    b: &cuda_core::DeviceBuffer<f32>,
    stream: &CudaStream,
) -> AppResult {
    let v = b.to_host_vec(stream)?;
    assert!(v.iter().all(|x| x.is_finite()), "nonfinite probe {name}");
    let sum = v.iter().map(|x| *x as f64).sum::<f64>();
    let sq = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
    let max = v.iter().fold(0.0_f32, |a, x| a.max(x.abs()));
    println!(
        "NUMERIC_F32 name={name} count={} hash={:016x} sum={sum:.17e} sumsq={sq:.17e} max={max:.9e}",
        v.len(),
        numeric_probe_hash(v.into_iter().map(f32::to_bits))
    );
    Ok(())
}

fn numeric_probe_attention_probs(
    name: &str,
    b: &cuda_core::DeviceBuffer<u16>,
    stream: &CudaStream,
) -> AppResult {
    let values = b.to_host_vec(stream)?;
    let seq = GPT2_SEQ_LEN;
    assert_eq!(values.len(), GPT2_BATCH_SIZE * GPT2_N_HEAD * seq * seq);
    let canonical = (0..GPT2_BATCH_SIZE * GPT2_N_HEAD).flat_map(|bh| {
        let values = &values;
        (0..seq).flat_map(move |query| {
            (0..=query).map(move |key| u32::from(values[(bh * seq + query) * seq + key]))
        })
    });
    println!(
        "ATTENTION_PROBS name={name} count={} hash={:016x}",
        GPT2_BATCH_SIZE * GPT2_N_HEAD * seq * (seq + 1) / 2,
        numeric_probe_hash(canonical)
    );
    Ok(())
}

#[test]
#[ignore = "requires the intact 1B FineWeb model; correctness fingerprints only"]
fn full_model_first_batch_numeric_fingerprints() -> AppResult {
    assert_eq!(
        (
            GPT2_BATCH_SIZE,
            GPT2_SEQ_LEN,
            GPT2_N_LAYER,
            GPT2_N_EMBD,
            GPT2_N_HEAD
        ),
        (4, 2048, 16, 2048, 32)
    );
    let mut trainer = Trainer::new(0x47505432)?;
    let mut data = TokenDataLoader::from_fineweb()?;
    let first = data.next_batch()?;
    let batch = trainer.batch_from_windows(&first.tokens, first.batch_size)?;
    trainer.materialize_training_weights()?;
    let stream = trainer.runtime.stream.as_ref();
    println!(
        "NUMERIC_INPUT tokens={:016x} targets={:016x}",
        numeric_probe_hash(batch.tokens.to_host_vec(stream)?),
        numeric_probe_hash(batch.targets.to_host_vec(stream)?)
    );
    for (name, t) in tensors(&trainer.uploaded) {
        let a = t.bytes.to_host_vec(stream)?;
        let b = t.scales.to_host_vec(stream)?;
        let g = t.global_scale.to_host_vec(stream)?;
        println!(
            "NUMERIC_WEIGHT name={name} payload={:016x} scales={:016x} global={:08x}",
            numeric_probe_hash(a.into_iter().map(u32::from)),
            numeric_probe_hash(b.into_iter().map(u32::from)),
            g[0].to_bits()
        );
    }
    trainer.forward_step(&batch)?;
    {
        let stream = trainer.runtime.stream.as_ref();
        let saved = trainer.buffers.tape.saved(
            &batch.tokens,
            batch.batch_size as u32,
            batch.seq_len as u32,
            batch.token_count as u32,
            &trainer.buffers.logits,
        );
        for (i, b) in saved.blocks.iter().enumerate() {
            if let Some(probs) = b.attention_probs {
                numeric_probe_attention_probs(&format!("blocks.{i}"), probs, stream)?;
            }
            numeric_probe_u16(&format!("blocks.{i}.qkv"), b.qkv, stream)?;
            numeric_probe_u16(
                &format!("blocks.{i}.attention_out"),
                b.attention_out,
                stream,
            )?;
            for (name, t) in [
                ("vnew", b.kda_v_new),
                ("inverse", b.kda_akk_inv),
                ("w", b.kda_w),
                ("aqk", b.kda_aqk),
            ] {
                if let Some(t) = t {
                    numeric_probe_u16(&format!("blocks.{i}.{name}"), t, stream)?;
                }
            }
        }
        numeric_probe_f32("final_normalized", &trainer.buffers.normalized, stream)?;
    }
    {
        let stream = trainer.runtime.stream.as_ref();
        let scratch = trainer.buffers.scratch.attention_core.args();
        numeric_probe_attention_probs("last_block", scratch.tc.p_half, stream)?;
    }
    trainer.enqueue_numeric_probe_backward(&batch)?;
    let stream = trainer.runtime.stream.as_ref();
    let grads = &trainer.buffers.backward;
    numeric_probe_f32("losses", &grads.losses, stream)?;
    numeric_probe_f32("lm_head_gradient", &grads.d_lm_head_weight, stream)?;
    numeric_probe_f32(
        "embedding_residual_gradient",
        &grads.d_embedding_residual,
        stream,
    )?;
    numeric_probe_f32("xsa_alphas_gradient", &grads.d_xsa_alphas, stream)?;
    for (i, b) in grads.blocks.iter().enumerate() {
        for (name, t) in [
            ("qkv_weight", &b.d_attn_qkv_weight),
            ("qkv_bias", &b.d_attn_qkv_bias),
            ("qk_scale", &b.d_attn_qk_scale),
            ("cproj_weight", &b.d_attn_c_proj_weight),
            ("cproj_bias", &b.d_attn_c_proj_bias),
            ("mlp_up_weight", &b.d_mlp_c_fc_weight),
            ("mlp_down_weight", &b.d_mlp_c_proj_weight),
            ("canon_a", &b.d_canon_a_weight),
            ("canon_c", &b.d_canon_c_weight),
        ] {
            numeric_probe_f32(&format!("blocks.{i}.{name}"), t, stream)?;
        }
    }
    assert_eq!(
        trainer.buffers.optimizer_state.next_step(),
        1,
        "the numerical probe must not update weights"
    );
    Ok(())
}

#[test]
#[ignore = "intact 1B repeated-forward scratch ownership counterexample; no optimizer updates"]
fn full_model_kda_scratch_preserves_probability_boundaries() -> AppResult {
    assert_eq!(
        (
            GPT2_BATCH_SIZE,
            GPT2_SEQ_LEN,
            GPT2_N_LAYER,
            GPT2_N_EMBD,
            GPT2_N_HEAD
        ),
        (4, 2048, 16, 2048, 32)
    );
    fn boundary_nonzeros(trainer: &mut Trainer, stage: &str) -> AppResult<usize> {
        let values = {
            let stream = trainer.runtime.stream.as_ref();
            let scratch = trainer.buffers.scratch.attention_core.args();
            scratch.tc.p_half.to_host_vec(stream)?
        };
        let seq = GPT2_SEQ_LEN;
        let window = gpt2_nvfp4::GPT2_FULL_ATTENTION_WINDOW;
        assert_eq!(values.len(), GPT2_BATCH_SIZE * GPT2_N_HEAD * seq * seq);
        let mut count = 0;
        let mut first = None;
        for bh in 0..GPT2_BATCH_SIZE * GPT2_N_HEAD {
            for query in 0..seq {
                let first_key = (query + 1).saturating_sub(window);
                let left_tile = (first_key / 64) * 64;
                let right_tile_end = ((query / 64 + 1) * 64).min(seq);
                for key in (left_tile..first_key).chain(query + 1..right_tile_end) {
                    let bits = values[(bh * seq + query) * seq + key];
                    if bits & 0x7fff != 0 {
                        count += 1;
                        first.get_or_insert((bh, query, key, bits));
                    }
                }
            }
        }
        println!("PROBABILITY_BOUNDARY stage={stage} nonzero={count} first={first:?}");
        Ok(count)
    }
    let mut trainer = Trainer::new(0x47505432)?;
    let mut data = TokenDataLoader::from_fineweb()?;
    let first = data.next_batch()?;
    let batch = trainer.batch_from_windows(&first.tokens, first.batch_size)?;
    trainer.materialize_training_weights()?;
    trainer.forward_step(&batch)?;
    assert_eq!(boundary_nonzeros(&mut trainer, "first_forward")?, 0);
    trainer.enqueue_numeric_probe_backward(&batch)?;
    assert_eq!(boundary_nonzeros(&mut trainer, "after_backward")?, 0);
    trainer.materialize_training_weights()?;
    trainer.forward_step(&batch)?;
    assert_eq!(
        boundary_nonzeros(&mut trainer, "second_forward")?,
        0,
        "KDA scratch must not contaminate masked full-attention probability tile mass on the next forward"
    );
    assert_eq!(trainer.buffers.optimizer_state.next_step(), 1);
    Ok(())
}
