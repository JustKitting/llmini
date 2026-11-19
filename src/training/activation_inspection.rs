//! Inspect values produced by the real CUDA forward; host arithmetic only summarizes tapes.
use super::{TokenDataLoader, Trainer};
use crate::{AppResult, reference_validation};
use gpt2_nvfp4::{
    AttentionDims, GPT2_ADAPTIVE_RELU2, GPT2_BOUNDED_SWIGLU, GPT2_MLP, GPT2_MLP_UP, GPT2_N_EMBD,
    GPT2_SEQ_LEN, MlpWeights, uses_full_attention,
};
use rust_kernels_cuda::{activation::RELU2_INITIAL_MEAN, nvfp4::csf};
use serde_json::{Value, json};
use std::{fs, path::Path};
mod decay;
mod engram;
mod kda_output;
mod mhc;

#[derive(Default)]
struct Moments {
    n: usize,
    nonfinite: usize,
    zeros: usize,
    sum: f64,
    square: f64,
    amax: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.n += 1;
        if !x.is_finite() {
            self.nonfinite += 1;
            return;
        }
        self.zeros += usize::from(x == 0.0);
        self.sum += x;
        self.square += x * x;
        self.amax = self.amax.max(x.abs());
    }
    fn json(&self) -> Value {
        let n = (self.n - self.nonfinite).max(1) as f64;
        let rms = (self.square / n).sqrt();
        json!({"values": self.n, "nonfinite": self.nonfinite,
            "mean": self.sum/n, "rms": rms, "max_abs": self.amax,
            "max_over_rms": self.amax/rms.max(f64::MIN_POSITIVE),
            "zero_fraction": self.zeros as f64/n})
    }
}

pub(super) fn run(
    checkpoint: &Path,
    output: Option<&Path>,
    replay_record: Option<&Path>,
    inspect_kda_decay: bool,
) -> AppResult {
    if !checkpoint.is_file() {
        return Err(format!("checkpoint does not exist: {}", checkpoint.display()).into());
    }
    let checkpoint_sha = reference_validation::sha256(checkpoint)?;
    let validation = TokenDataLoader::fineweb_validation_tokens()?;
    let (expected_ce, verified_replay_record) =
        super::reference_eval::replay_expectation(checkpoint, replay_record)?;
    let output = output.map(Path::to_path_buf).unwrap_or_else(|| {
        checkpoint
            .parent()
            .unwrap_or(Path::new("."))
            .join("activation_inspection.json")
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let quantization = crate::checkpoint::audit_model(checkpoint)?;
    let audit_path = output.with_extension("quantization.json");
    fs::write(
        &audit_path,
        serde_json::to_string_pretty(&quantization)? + "\n",
    )?;
    let mut trainer = Trainer::new(0x47505432)?;
    trainer.load_model(checkpoint)?;
    let batch = trainer.batch_from_windows(&validation, super::data::VALIDATION_WINDOWS)?;
    let before = trainer.eval_token_losses(&batch)?;
    let native_ce = before.iter().sum::<f32>() / before.len() as f32;
    if before.iter().any(|x| !x.is_finite())
        || expected_ce.is_some_and(|x| (x - f64::from(native_ce)).abs() > 1e-4)
    {
        return Err("native checkpoint CE does not match the recorded run".into());
    }
    let stream = trainer.runtime.stream.as_ref();
    let tape = trainer.buffers.tape.saved(
        &batch.tokens,
        batch.batch_size as u32,
        batch.seq_len as u32,
        batch.token_count as u32,
        &trainer.buffers.logits,
    );
    let rows = (0..batch.batch_size)
        .flat_map(|sequence| {
            (0..8).map(move |i| sequence * GPT2_SEQ_LEN + (2 * i + 1) * GPT2_SEQ_LEN / 16)
        })
        .collect::<Vec<_>>();
    let mut layers = Vec::new();
    let mut decay_inspector = inspect_kda_decay
        .then(|| decay::Inspector::new(&trainer, batch.token_count))
        .transpose()?;
    let mut decay_ok = true;
    let mut decay_layers = 0;
    let mut nonfinite = quantization["nonfinite_values"].as_u64().unwrap_or(1) as usize;
    let mut descriptors =
        rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors::new(stream)?;
    for (index, block) in tape.blocks.iter().enumerate() {
        let decay_report = if let Some(inspector) = decay_inspector
            .as_mut()
            .filter(|_| !uses_full_attention(index))
        {
            let report = inspector.inspect(
                &trainer.runtime,
                block.qkv_input_nvfp4,
                trainer.uploaded.blocks[index].attention_tensors(
                    trainer.uploaded.xsa_alphas.device(),
                    (index * gpt2_nvfp4::GPT2_N_HEAD) as u32,
                ),
                block.qkv,
                batch.token_count as u32,
                batch.seq_len as u32,
            )?;
            decay_ok &= report["passed"] == true;
            decay_layers += 1;
            Some(report)
        } else {
            None
        };
        let paired = if GPT2_BOUNDED_SWIGLU {
            MlpWeights::project_paired(
                &trainer.runtime.tma_gemm,
                stream,
                block.mlp_up_input_nvfp4,
                trainer.uploaded.blocks[index].mlp_tensors().up,
                &mut descriptors,
                &mut trainer.buffers.scratch.mlp_paired,
                batch.token_count as u32,
            )?;
            trainer.runtime.mlp.bounded_forward(
                stream,
                &trainer.buffers.scratch.mlp_paired,
                &mut trainer.buffers.mlp_act,
                batch.token_count as u32,
                GPT2_MLP as u32,
            )?;
            Some(trainer.buffers.scratch.mlp_paired.to_host_vec(stream)?)
        } else {
            None
        };
        let pre = if GPT2_BOUNDED_SWIGLU {
            Vec::new()
        } else {
            block.mlp_up.to_host_vec(stream)?
        };
        let packed = block.mlp_down_input_nvfp4.bytes.to_host_vec(stream)?;
        let raw_scales = block.mlp_down_input_nvfp4.scales.to_host_vec(stream)?;
        let globals = block
            .mlp_down_input_nvfp4
            .global_scales
            .to_host_vec(stream)?;
        let scales = rowwise_scale_header(GPT2_MLP, globals.len(), &raw_scales)?;
        let exact_last = if GPT2_BOUNDED_SWIGLU || index + 1 == tape.blocks.len() {
            Some(trainer.buffers.mlp_act.to_host_vec(stream)?)
        } else {
            None
        };
        let mut pre_stats = Moments::default();
        let mut up_stats = Moments::default();
        let mut activation = Moments::default();
        let mut represented = Moments::default();
        let mut error = Moments::default();
        let mut exact_error = Moments::default();
        let mut exact_activation = Moments::default();
        let mut zero_scales = 0;
        for &row in &rows {
            let relu2_shift = if GPT2_ADAPTIVE_RELU2 {
                pre[row * GPT2_MLP..(row + 1) * GPT2_MLP]
                    .iter()
                    .map(|&bits| {
                        let p = f16(bits).max(0.0);
                        p * p
                    })
                    .sum::<f32>()
                    / GPT2_MLP as f32
            } else {
                RELU2_INITIAL_MEAN
            };
            for group in 0..GPT2_MLP / 16 {
                let code = csf::scale_at(&scales, row * (GPT2_MLP / 16) + group);
                let scale = f64::from(globals[row]) * fp8(code);
                zero_scales += usize::from(scale == 0.0);
                for lane in 0..16 {
                    let position = row * GPT2_MLP + group * 16 + lane;
                    let (p, value) = if let Some(paired) = &paired {
                        let gate = paired[row * GPT2_MLP_UP + group * 16 + lane];
                        let up = paired[row * GPT2_MLP_UP + GPT2_MLP + group * 16 + lane];
                        up_stats.add(f64::from(up));
                        (
                            gate,
                            f64::from(
                                exact_last.as_ref().expect("native bounded activation")[position],
                            ),
                        )
                    } else {
                        let p = f16(pre[position]);
                        let relu = p.max(0.0);
                        (p, f64::from(relu * relu - relu2_shift))
                    };
                    let byte = packed[position / 2];
                    let code = if position % 2 == 0 {
                        byte & 15
                    } else {
                        byte >> 4
                    };
                    let quantized = fp4(code) * scale;
                    pre_stats.add(f64::from(p));
                    activation.add(value);
                    represented.add(quantized);
                    error.add(quantized - value);
                    if let Some(exact) = &exact_last {
                        exact_activation.add(f64::from(exact[position]));
                        exact_error.add(quantized - f64::from(exact[position]));
                    }
                }
            }
        }
        let qkv = block.qkv.to_host_vec(stream)?;
        let dims = AttentionDims::new(uses_full_attention(index));
        let mut norms: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for &row in &rows {
            for head in 0..dims.head_count as usize {
                for (part, values) in norms.iter_mut().enumerate() {
                    let base = row * dims.qkv_dim as usize
                        + part * GPT2_N_EMBD
                        + head * dims.head_dim as usize;
                    let square = qkv[base..base + dims.head_dim as usize]
                        .iter()
                        .map(|&x| f64::from(f16(x)).powi(2))
                        .sum::<f64>();
                    let norm = square.sqrt();
                    nonfinite += usize::from(!norm.is_finite());
                    values.push(norm);
                }
            }
        }
        nonfinite += pre_stats.nonfinite
            + up_stats.nonfinite
            + activation.nonfinite
            + represented.nonfinite
            + error.nonfinite;
        let mut layer = json!({"layer": index,
            "attention": if uses_full_attention(index) {"full"} else {"kda"},
            "mlp_kind": gpt2_nvfp4::GPT2_MLP_KIND_NAME,
            "sampled_rows": rows.len(),
            "preactivation_fp16": (!GPT2_BOUNDED_SWIGLU).then(|| pre_stats.json()),
            "paired_gate_preactivation_fp32": GPT2_BOUNDED_SWIGLU.then(|| pre_stats.json()),
            "paired_up_preactivation_fp32": GPT2_BOUNDED_SWIGLU.then(|| up_stats.json()),
            "activation_reconstructed_from_fp16": (!GPT2_BOUNDED_SWIGLU).then(|| activation.json()),
            "activation_native_fp32": GPT2_BOUNDED_SWIGLU.then(|| activation.json()),
            "activation_nvfp4": represented.json(), "approximate_quantization_error": error.json(),
            "approximate_relative_l2_error": (error.square/activation.square.max(f64::MIN_POSITIVE)).sqrt(),
            "mean_error_over_activation_rms": (error.sum/error.n as f64)/(activation.square/activation.n as f64).sqrt().max(f64::MIN_POSITIVE),
            "zero_local_scale_groups": zero_scales,
            "exact_last_layer_relative_l2_error": exact_last.as_ref().map(|_| (exact_error.square/exact_activation.square.max(f64::MIN_POSITIVE)).sqrt()),
            "q_head_norm": distribution(&mut norms[0]),
            "k_head_norm": distribution(&mut norms[1]),
            "v_head_norm": distribution(&mut norms[2]),
            "v_tape_coordinate":if gpt2_nvfp4::GPT2_VALUE_RMS {"post_effective_head_RMS_pre_beta"} else {"mixed_raw_V_before_KDA_SiLU_beta"},
            "value_rms_epsilon":gpt2_nvfp4::GPT2_VALUE_RMS.then_some(1e-6),
        });
        if let Some(report) = decay_report {
            layer["kda_decay_clamp"] = report;
        }
        if !uses_full_attention(index) {
            layer["kda_output"] = kda_output::inspect(stream, block, &qkv, &rows)?;
        }
        println!("activation_inspection_layer={layer}");
        layers.push(layer);
    }
    let (engram, engram_nonfinite) = engram::inspect(&trainer, &rows)?;
    nonfinite += engram_nonfinite;
    let (mhc_head, mhc_nonfinite) = mhc::inspect(&trainer, &rows)?;
    nonfinite += mhc_nonfinite;
    let after = trainer.eval_token_losses(&batch)?;
    let replay_exact = before == after;
    let unchanged = trainer.buffers.optimizer_state.step == 0;
    let expected_decay_layers = (0..gpt2_nvfp4::GPT2_N_LAYER)
        .filter(|&i| !uses_full_attention(i))
        .count();
    let checkpoint_unchanged =
        !inspect_kda_decay || reference_validation::sha256(checkpoint)? == checkpoint_sha;
    let passed = replay_exact
        && unchanged
        && nonfinite == 0
        && decay_ok
        && checkpoint_unchanged
        && (!inspect_kda_decay || decay_layers == expected_decay_layers);
    let mut result = json!({"passed": passed, "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha, "native_ce": native_ce,
        "recorded_native_ce": expected_ce, "batch_size": batch.batch_size,
        "producer_recorded_native_ce":super::reference_eval::expected_native_ce(checkpoint)?,
        "verified_replay_provenance":verified_replay_record,
        "context_tokens": batch.seq_len, "selected_token_rows": rows,
        "selection": "eight equidistant token-position midpoints per native validation sequence; all features",
        "optimizer_updates": trainer.buffers.optimizer_state.step,
        "ce_replay_bit_exact": replay_exact, "nonfinite_sampled_values": nonfinite,
        "checkpoint_quantization_audit": audit_path, "layers": layers, "engram": engram,
        "mhc_head": mhc_head,
        "mlp_kind": if GPT2_BOUNDED_SWIGLU {"bounded-swiglu"} else {"relu2"},
        "mlp_activation_source": if GPT2_BOUNDED_SWIGLU {"exact_native_NVFP4_paired_projection_recomputation_and_native_bounded_activation_all_layers"} else {"rounded_FP16_preactivation_reconstruction_with_original_FP32_final_layer"},
        "limits": "CUDA forward and, for bounded-SwiGLU, exact native paired projection recomputation only. Host decoding and reductions inspect actual values; no CPU training or matrix products. ReLU activation reconstruction uses rounded FP16 preactivation and its error is approximate. Q/K/V norms are from saved pre-mixer tapes, before QK normalization and KDA feature activations. A fixed final-checkpoint sample cannot establish causality for an unsaved earlier state.",
    });
    if inspect_kda_decay {
        result["kda_decay_audit"] = json!({"passed":decay_ok && decay_layers==expected_decay_layers,
        "measured_layers":decay_layers,"expected_layers":expected_decay_layers,"checkpoint_unchanged":checkpoint_unchanged,
        "forward_passes":if trainer.latent_feedback.is_some() {super::latent_feedback::EVAL_PASSES}else{1},
        "coverage":"all decay coordinates in the complete native validation batch; not the midpoint-only norm sample",
        "CPU_training":false,"optimizer_updates":0,
        "limits":"A final-checkpoint evaluation-pass sample does not establish exposure during original training or causal impact on loss."});
    }
    fs::write(&output, serde_json::to_string_pretty(&result)? + "\n")?;
    println!(
        "activation_inspection_result={} passed={passed}",
        output.display()
    );
    if !passed {
        return Err("activation inspection failed; details saved".into());
    }
    Ok(())
}

fn rowwise_scale_header(width: usize, rows: usize, raw: &[u8]) -> AppResult<Vec<u8>> {
    if raw.len() != csf::rowwise_encoded_len(rows, width) {
        return Err("unexpected rowwise CSF activation shape".into());
    }
    let mut scales = vec![0_u8; csf::HEADER_BYTES];
    scales[..4].copy_from_slice(&(width as u32).to_le_bytes());
    scales[4..8].copy_from_slice(&(rows as u32).to_le_bytes());
    scales[8] = (width / 16).trailing_zeros() as u8;
    scales[9] = 7;
    scales.extend_from_slice(raw);
    Ok(scales)
}

fn distribution(values: &mut [f64]) -> Value {
    values.sort_by(f64::total_cmp);
    json!({"heads": values.len(), "min": values[0],
        "zero_heads": values.iter().filter(|&&value| value==0.0).count(),
        "median": values[values.len()/2],
        "p99": values[(values.len()-1)*99/100], "max": values[values.len()-1]})
}
fn fp4(code: u8) -> f64 {
    let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code & 7)];
    if code & 8 == 0 { value } else { -value }
}
fn fp8(code: u8) -> f64 {
    let exponent = (code >> 3) & 15;
    let fraction = code & 7;
    let value = match exponent {
        0 => f64::from(fraction) / 512.0,
        15 if fraction == 7 => f64::NAN,
        _ => (1.0 + f64::from(fraction) / 8.0) * 2.0_f64.powi(i32::from(exponent) - 7),
    };
    if code & 128 == 0 { value } else { -value }
}
fn f16(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let value = match exponent {
        0 => f32::from(fraction) / 16_777_216.0,
        31 if fraction == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => {
            f32::from_bits((u32::from(exponent) + 112) << 23) * (1.0 + f32::from(fraction) / 1024.0)
        }
    };
    if bits & 0x8000 == 0 { value } else { -value }
}
