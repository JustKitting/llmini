//! End-to-end native value-RMS -> attention -> XSA -> reverse-chain check.
//! Failure cases were declared in value_normalization_native_contract_20261115.md:
//! beta outside RMS, beta adjoint, absence of second SiLU, cached/uncached
//! preparation, normalized-coordinate XSA, one shared complete RMS reverse.
use crate::{AppResult, training::runtime::Runtime};
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, AttentionDims, GPT2_BATCH_SIZE as B, GPT2_N_EMBD as D,
    GPT2_N_HEAD as HEADS, GPT2_SEQ_LEN as S, attention_headwise_gate_offset,
};
use rust_kernels_cuda::{
    attention::{
        ApplyRopeArgs, CausalAttentionBackwardTcArgs, CausalAttentionTcArgs,
        ExclusiveSelfAttentionBackwardArgs, ExclusiveSelfAttentionForwardArgs,
    },
    nvfp4::{Nvfp4DeviceTensor, csf},
};
#[path = "../../../crates/cuda-kernels/tests/common/f16_round.rs"]
mod half_input;

const H: usize = 64;
const N: usize = B * S;
fn pattern(j: usize) -> f64 {
    [0.125, -0.25, 0.5, -0.125][j % 4]
}
fn incoming(j: usize) -> f64 {
    [0.25, 0.125, -0.5, 0.0625][j % 4]
}
fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}
fn raw_value(batch: usize, head: usize, j: usize) -> f32 {
    (pattern(j + head) * [0.001, 1., 32., 1e6][batch % 4]) as f32
}
fn beta(head: usize) -> f64 {
    [-2., 0., 2.][head % 3]
}
fn close(a: f32, e: f64, label: &str, tol: f64) {
    let error = (f64::from(a) - e).abs();
    assert!(
        a.is_finite() && error <= tol * (e.abs() + 1e-5),
        "{label} native={a:e} oracle={e:e} error={error:e}"
    );
}

#[test]
#[ignore = "actual B8/S2048/d2048 RMS/full-KDA/XSA chain and beta adjoints; zero updates"]
fn native_value_rms_attention_XSA_beta_complete_chain() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_VALUE_RMS);
    let runtime = Runtime::new()?;
    let stream = runtime.stream.as_ref();
    let mut scratch = AttentionCoreScratchBuffers::new(stream)?;
    let payload = DeviceBuffer::from_host(stream, &vec![0x22u8; 64])?;
    let scales = DeviceBuffer::from_host(stream, &csf::encode_host(&[0x38; 8], 1, 128))?;
    let alpha_scale = DeviceBuffer::from_host(stream, &[0.25f32])?;
    let qk_scale = DeviceBuffer::from_host(stream, &[1f32])?;
    let alpha = Nvfp4DeviceTensor::new(&payload, &scales, &alpha_scale);
    let qk = Nvfp4DeviceTensor::new(&payload, &scales, &qk_scale);
    for full in [false, true] {
        let dims = AttentionDims::new(full);
        let stride = dims.qkv_dim as usize;
        let gate = attention_headwise_gate_offset(full);
        let mut values = vec![0f32; N * stride];
        let mut upstream = vec![0f32; N * D];
        for row in 0..N {
            for head in 0..HEADS {
                values[row * stride + head * H] = 1.;
                values[row * stride + D + head * H] = 1.;
                for j in 0..H {
                    values[row * stride + 2 * D + head * H + j] = raw_value(row / S, head, j);
                    if !full {
                        values[row * stride + 3 * D + head * H + j] = -3.;
                    }
                    if row % S == 0 {
                        upstream[row * D + head * H + j] = incoming(j) as f32;
                    }
                }
                if !full {
                    values[row * stride + 4 * D + head] = beta(head) as f32;
                }
                values[row * stride + gate + head] = 0.5;
            }
        }
        let raw = DeviceBuffer::from_host(stream, &values)?;
        let mut qkv = DeviceBuffer::from_host(stream, &values)?;
        drop(values);
        runtime
            .value_rms
            .forward(stream, &mut qkv, N as u32, D as u32, stride as u32, !full)?;
        let mut tape = DeviceBuffer::<u16>::zeroed(stream, N * stride)?;
        if full {
            runtime.attention.apply_rope(ApplyRopeArgs {
                stream,
                qkv: &mut qkv,
                qkv_f16: Some(&mut tape),
                row_count: N as u32,
                seq_len: S as u32,
                batch_size: B as u32,
                embedding_dim: D as u32,
                qkv_dim: stride as u32,
                head_count: HEADS as u32,
                head_dim: H as u32,
            })?;
        }
        let mut states = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut xsa_raw = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut output = DeviceBuffer::<f32>::zeroed(stream, N * D)?;
        let mut lse = DeviceBuffer::<f32>::zeroed(stream, N * HEADS)?;
        let mut probabilities =
            DeviceBuffer::<u16>::zeroed(stream, if full { B * HEADS * S * S } else { 1 })?;
        let mut vnew = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut inverse = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut w = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut aqk = DeviceBuffer::<u16>::zeroed(stream, N * D)?;
        let mut first_output = None;
        for cached in [true, false] {
            let args = CausalAttentionTcArgs {
                stream,
                tc_module: &runtime.f16_tc_matmul,
                qkv: &qkv,
                qk_scale: qk,
                out: &mut output,
                qkv_f16: cached.then_some(&mut tape),
                attention_out_f16: Some(&mut states),
                forward_probs_f16: full.then_some(&mut probabilities),
                kda_v_new: (!full && cached).then_some(&mut vnew),
                kda_akk_inv: (!full && cached).then_some(&mut inverse),
                kda_w: (!full && cached).then_some(&mut w),
                kda_aqk: (!full && cached).then_some(&mut aqk),
                log_sum_exp: &mut lse,
                scratch: scratch.forward_tc(),
                row_count: N as u32,
                seq_len: S as u32,
                batch_size: B as u32,
                embedding_dim: D as u32,
                qkv_dim: stride as u32,
                head_count: HEADS as u32,
                head_dim: H as u32,
                attention_window: S as u32,
                partial_key_offset: false,
                selective_attention: false,
                stable_mask_gamma: 0.,
            };
            if full {
                runtime.attention.causal_attention_tc(args)?;
            } else {
                runtime.attention.kda_attention_tc(args)?;
            }
            let actual = output.to_host_vec(stream)?;
            if let Some(first) = first_output.as_ref() {
                assert_eq!(first, &actual, "cached/uncached full forward mismatch");
            } else {
                first_output = Some(actual);
            }
        }
        runtime
            .attention
            .exclusive_self_attention_forward(ExclusiveSelfAttentionForwardArgs {
                stream,
                qkv: &qkv,
                out: &mut output,
                xsa_alphas: alpha,
                qkv_f16: Some(&mut tape),
                raw_out_f16: Some(&mut xsa_raw),
                row_count: N as u32,
                embedding_dim: D as u32,
                qkv_dim: stride as u32,
                head_count: HEADS as u32,
                head_dim: H as u32,
                value_offset: (2 * D) as u32,
                gate_offset: gate as u32,
                alpha_offset: 0,
                apply_headwise_gate: true,
                kda_value_activation: false,
            })?;
        let actual = output.to_host_vec(stream)?;
        let normalized_source = qkv.to_host_vec(stream)?;
        let actual_tape = tape.to_host_vec(stream)?;
        for backward_cached in if full {
            &[true][..]
        } else {
            &[true, false][..]
        } {
            let mut dqkv = DeviceBuffer::<f32>::zeroed(stream, N * stride)?;
            let mut amax = DeviceBuffer::<f32>::zeroed(stream, 5 * N)?;
            let mut d_alpha = DeviceBuffer::<f32>::zeroed(stream, 128)?;
            let mut d_scale = DeviceBuffer::<f32>::zeroed(stream, 128)?;
            let mut d_out = DeviceBuffer::from_host(stream, &upstream)?;
            runtime.attention.exclusive_self_attention_backward(
                ExclusiveSelfAttentionBackwardArgs {
                    stream,
                    qkv_f16: &tape,
                    raw_out_f16: &xsa_raw,
                    d_out: &mut d_out,
                    d_qkv: &mut dqkv,
                    d_qkv_chunk_amax: &mut amax,
                    d_xsa_alphas: &mut d_alpha,
                    xsa_alphas: alpha,
                    row_count: N as u32,
                    embedding_dim: D as u32,
                    qkv_dim: stride as u32,
                    head_count: HEADS as u32,
                    head_dim: H as u32,
                    value_offset: (2 * D) as u32,
                    gate_offset: gate as u32,
                    gate_amax_offset: if full { 3 * N as u32 } else { 4 * N as u32 },
                    alpha_offset: 0,
                    apply_headwise_gate: true,
                    kda_value_activation: false,
                },
            )?;
            let core = scratch.args();
            let args = CausalAttentionBackwardTcArgs {
                stream,
                tc_module: &runtime.f16_tc_matmul,
                qkv: &tape,
                attention_out: &states,
                kda_v_new: (!full && *backward_cached).then_some(&vnew),
                kda_akk_inv: (!full && *backward_cached).then_some(&inverse),
                kda_w: (!full && *backward_cached).then_some(&w),
                kda_aqk: (!full && *backward_cached).then_some(&aqk),
                d_out: &d_out,
                qk_scale: qk,
                log_sum_exp: &lse,
                softmax_d: core.softmax_d,
                qk_norm_max: core.qk_norm_max,
                d_qkv: &mut dqkv,
                d_qkv_chunk_amax: &mut amax,
                d_qk_scale: &mut d_scale,
                accumulate_value_grad: true,
                scratch: core.tc,
                row_count: N as u32,
                seq_len: S as u32,
                batch_size: B as u32,
                embedding_dim: D as u32,
                qkv_dim: stride as u32,
                head_count: HEADS as u32,
                head_dim: H as u32,
                attention_window: S as u32,
                partial_key_offset: false,
                selective_attention: false,
                qk_norm_offset: 0,
                backward_mask_seed: 0,
                backward_tile_budget: 0.,
                reuse_forward_probs: full,
                forward_probs_f16: full.then_some(&probabilities),
            };
            if full {
                runtime.attention.causal_attention_backward_tc(args)?;
            } else {
                runtime.attention.kda_attention_backward_tc(args)?;
            }
            runtime.value_rms.backward(
                stream,
                &raw,
                &mut dqkv,
                N as u32,
                D as u32,
                stride as u32,
                !full,
            )?;
            let gradients = dqkv.to_host_vec(stream)?;
            assert!(gradients.iter().all(|x| x.is_finite()));
            let mut beta_nonzero = 0;
            let mut output_by_beta = [0f64; 3];
            for batch in 0..B {
                for head in 0..HEADS {
                    let row = batch * S;
                    let vb = row * stride + 2 * D + head * H;
                    let ob = row * D + head * H;
                    let effective = (0..H)
                        .map(|j| {
                            let x = f64::from(raw_value(batch, head, j));
                            if full { x } else { x * sigmoid(x) }
                        })
                        .collect::<Vec<_>>();
                    let square = effective.iter().map(|x| x * x).sum::<f64>() / H as f64;
                    let inv = (square + f64::from(1e-6f32)).sqrt().recip();
                    let normalized = effective.iter().map(|v| v * inv).collect::<Vec<_>>();
                    let represented = (0..H)
                        .map(|j| {
                            let stored = half_input::decode(actual_tape[vb + j]);
                            assert_eq!(
                                stored.to_bits(),
                                half_input::round(normalized_source[vb + j]).to_bits(),
                                "actual forward normalized-V tape must be FP16"
                            );
                            f64::from(stored)
                        })
                        .collect::<Vec<_>>();
                    let direction_square =
                        represented.iter().map(|v| v * v).sum::<f64>() / H as f64;
                    let dot = (0..H).map(|j| incoming(j) * represented[j]).sum::<f64>() / H as f64;
                    let q = sigmoid(1.);
                    let c = if full {
                        1.
                    } else {
                        0.125 * (q / ((q * q + 1e-6).sqrt() + 1e-6)).powi(2)
                    };
                    let b = if full { 1. } else { sigmoid(beta(head)) };
                    let factor = c * b * (1. - 0.25f64.tanh()) * sigmoid(0.5);
                    for j in 0..H {
                        close(
                            actual[ob + j],
                            factor * normalized[j],
                            &format!(
                                "complete chain forward full={full} batch={batch} head={head} j={j}"
                            ),
                            0.004,
                        );
                        let x = f64::from(half_input::round(raw_value(batch, head, j)));
                        let s = sigmoid(x);
                        let activation_grad = if full || x == f64::INFINITY {
                            1.
                        } else if x == f64::NEG_INFINITY {
                            0.
                        } else {
                            s + x * s * (1. - s)
                        };
                        let projection = if direction_square > 0. {
                            represented[j] * dot / direction_square
                        } else {
                            0.
                        };
                        let radial_fraction = f64::from(1e-6f32) / (square + f64::from(1e-6f32));
                        let expected = factor
                            * inv
                            * ((incoming(j) - projection) + projection * radial_fraction)
                            * activation_grad;
                        close(
                            gradients[vb + j],
                            expected,
                            &format!(
                                "complete RMS/core/XSA raw-V reverse full={full} batch={batch} head={head} j={j}"
                            ),
                            0.012,
                        );
                    }
                    let loss = normalized
                        .iter()
                        .enumerate()
                        .map(|(j, v)| v * incoming(j))
                        .sum::<f64>();
                    if !full {
                        let expected =
                            c * b * (1. - b) * (1. - 0.25f64.tanh()) * sigmoid(0.5) * loss;
                        let db = gradients[row * stride + 4 * D + head];
                        close(
                            db,
                            expected,
                            "complete beta reverse uses normalized V",
                            0.012,
                        );
                        assert_ne!(db, 0.);
                        beta_nonzero += 1;
                        if batch == 0 && head < 3 {
                            output_by_beta[head] = actual[ob].abs() as f64 / normalized[0].abs();
                        }
                    }
                }
            }
            if !full {
                assert!(
                    output_by_beta[0] < output_by_beta[1] && output_by_beta[1] < output_by_beta[2],
                    "normalization cancelled beta strength: {output_by_beta:?}"
                );
            }
            let mut max_future = 0f32;
            for row in 0..N {
                if row % S != 0 {
                    for &x in &gradients[row * stride..(row + 1) * stride] {
                        max_future = max_future.max(x.abs());
                    }
                }
            }
            assert!(
                max_future <= 1e-5,
                "first-token objective backpropagated into future: {max_future:e}"
            );
            println!(
                "complete_value_RMS_attention_chain_verified B={B} S={S} d={D} full={full} backward_cached={backward_cached} cached_uncached_forward=bit_exact beta_nonzero_adjoints={beta_nonzero} beta_strength=retained XSA=no_second_SiLU combined_RMS_reverse=checked future_gradient_max={max_future:e} optimizer_updates=0"
            );
        }
    }
    Ok(())
}
