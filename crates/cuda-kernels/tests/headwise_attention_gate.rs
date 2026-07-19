use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{
    AttentionModule, ExclusiveSelfAttentionBackwardArgs, ExclusiveSelfAttentionForwardArgs,
    HeadwiseAttentionGateBackwardArgs, HeadwiseAttentionGateForwardArgs,
};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;

const ROWS: usize = 2;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const EMBEDDING: usize = HEADS * HEAD_DIM;
const QKV_DIM: usize = 30;
const GATE_OFFSET: usize = 24;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn headwise_gate_forward_and_backward_match_chain_rule() -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(AttentionModule::from_module)?;
    let mut qkv = vec![0.0_f32; ROWS * QKV_DIM];
    let gate_logits = [[-1.25_f32, 0.75_f32], [2.0_f32, -0.5_f32]];
    for row in 0..ROWS {
        for head in 0..HEADS {
            qkv[row * QKV_DIM + GATE_OFFSET + head] = gate_logits[row][head];
        }
    }
    let raw: Vec<f32> = (0..ROWS * EMBEDDING)
        .map(|index| -1.75 + index as f32 * 0.25)
        .collect();
    let qkv_dev = DeviceBuffer::from_host(&stream, &qkv)?;
    let mut out_dev = DeviceBuffer::from_host(&stream, &raw)?;
    let mut qkv_f16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * QKV_DIM)?;
    let mut raw_f16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * EMBEDDING)?;

    module.headwise_attention_gate_forward(HeadwiseAttentionGateForwardArgs {
        stream: &stream,
        qkv: &qkv_dev,
        out: &mut out_dev,
        qkv_f16: Some(&mut qkv_f16),
        raw_out_f16: Some(&mut raw_f16),
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        gate_offset: GATE_OFFSET as u32,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    for row in 0..ROWS {
        for head in 0..HEADS {
            let gate = sigmoid(gate_logits[row][head]);
            for dim in 0..HEAD_DIM {
                let index = row * EMBEDDING + head * HEAD_DIM + dim;
                common::assert_close(out[index], raw[index] * gate, 2.0e-6);
            }
        }
    }

    let upstream: Vec<f32> = (0..ROWS * EMBEDDING)
        .map(|index| 0.125 + index as f32 * 0.0625)
        .collect();
    let mut d_out = DeviceBuffer::from_host(&stream, &upstream)?;
    let sentinel = -7.0_f32;
    let mut d_qkv = DeviceBuffer::from_host(&stream, &vec![sentinel; ROWS * QKV_DIM])?;
    let mut d_qkv_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    module.headwise_attention_gate_backward(HeadwiseAttentionGateBackwardArgs {
        stream: &stream,
        qkv_f16: &qkv_f16,
        raw_out_f16: &raw_f16,
        d_out: &mut d_out,
        d_qkv: &mut d_qkv,
        d_qkv_chunk_amax: &mut d_qkv_chunk_amax,
        gate_amax_offset: 0,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        gate_offset: GATE_OFFSET as u32,
    })?;

    let d_out = d_out.to_host_vec(&stream)?;
    let d_qkv = d_qkv.to_host_vec(&stream)?;
    let saved_qkv = qkv_f16.to_host_vec(&stream)?;
    let saved_raw = raw_f16.to_host_vec(&stream)?;
    let gate_amax = d_qkv_chunk_amax.to_host_vec(&stream)?;
    for row in 0..ROWS {
        let mut expected_row_amax = 0.0_f32;
        for head in 0..HEADS {
            let gate_index = row * QKV_DIM + GATE_OFFSET + head;
            let gate = sigmoid(f16_bits_to_f32(saved_qkv[gate_index]));
            let mut dot = 0.0;
            for dim in 0..HEAD_DIM {
                let index = row * EMBEDDING + head * HEAD_DIM + dim;
                common::assert_close(d_out[index], upstream[index] * gate, 2.0e-6);
                dot += upstream[index] * f16_bits_to_f32(saved_raw[index]);
            }
            let expected_gate_grad = dot * gate * (1.0 - gate);
            common::assert_close(d_qkv[gate_index], expected_gate_grad, 2.0e-6);
            expected_row_amax = expected_row_amax.max(expected_gate_grad.abs());
        }
        common::assert_close(gate_amax[row], expected_row_amax, 2.0e-6);
        for col in 0..QKV_DIM {
            if (GATE_OFFSET..GATE_OFFSET + HEADS).contains(&col) {
                continue;
            }
            let expected = if col >= GATE_OFFSET + HEADS {
                0.0
            } else {
                sentinel
            };
            assert_eq!(d_qkv[row * QKV_DIM + col], expected);
        }
    }

    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gated_xsa_forward_and_backward_match_reference() -> Result<(), Box<dyn Error>> {
    run_xsa_case(false)?;
    run_xsa_case(true)
}

fn run_xsa_case(kda_value_activation: bool) -> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(AttentionModule::from_module)?;
    let mut qkv = vec![0.0_f32; ROWS * QKV_DIM];
    let gate_logits = [[-0.75_f32, 0.5_f32], [1.25_f32, -1.0_f32]];
    for row in 0..ROWS {
        for head in 0..HEADS {
            for dim in 0..HEAD_DIM {
                let value_index = row * QKV_DIM + 2 * EMBEDDING + head * HEAD_DIM + dim;
                qkv[value_index] = -1.0 + 0.1875 * (row * EMBEDDING + head * HEAD_DIM + dim) as f32;
            }
            qkv[row * QKV_DIM + GATE_OFFSET + head] = gate_logits[row][head];
        }
    }
    let raw: Vec<f32> = (0..ROWS * EMBEDDING)
        .map(|index| -0.875 + index as f32 * 0.125)
        .collect();
    let alpha_raw = [0.25_f32, -0.25_f32];
    let alpha_bytes = DeviceBuffer::from_host(&stream, &[0xa2, 0, 0, 0, 0, 0, 0, 0])?;
    let alpha_scales = DeviceBuffer::from_host(&stream, &[common::nvfp4::E4M3_ONE])?;
    let alpha_global_scale = DeviceBuffer::from_host(&stream, &[0.25_f32])?;
    let alpha_device = Nvfp4DeviceTensor::new(&alpha_bytes, &alpha_scales, &alpha_global_scale);

    let qkv_dev = DeviceBuffer::from_host(&stream, &qkv)?;
    let mut out_dev = DeviceBuffer::from_host(&stream, &raw)?;
    // The attention core saves QKV before XSA. Seed that tape here, then let
    // the fused XSA path overwrite the head-gate logits exactly as in training.
    let qkv_half: Vec<u16> = qkv.iter().copied().map(f32_to_f16_bits).collect();
    let mut qkv_f16 = DeviceBuffer::from_host(&stream, &qkv_half)?;
    let mut raw_f16 = DeviceBuffer::<u16>::zeroed(&stream, ROWS * EMBEDDING)?;
    module.exclusive_self_attention_forward(ExclusiveSelfAttentionForwardArgs {
        stream: &stream,
        qkv: &qkv_dev,
        out: &mut out_dev,
        xsa_alphas: alpha_device,
        qkv_f16: Some(&mut qkv_f16),
        raw_out_f16: Some(&mut raw_f16),
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        value_offset: (2 * EMBEDDING) as u32,
        gate_offset: GATE_OFFSET as u32,
        alpha_offset: 0,
        apply_headwise_gate: true,
        kda_value_activation,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    for row in 0..ROWS {
        for head in 0..HEADS {
            let a = alpha_raw[head].tanh();
            let gate = sigmoid(gate_logits[row][head]);
            let value_base = row * QKV_DIM + 2 * EMBEDDING + head * HEAD_DIM;
            let hidden_base = row * EMBEDDING + head * HEAD_DIM;
            let values: Vec<f32> = (0..HEAD_DIM)
                .map(|dim| xsa_test_value(qkv[value_base + dim], kda_value_activation))
                .collect();
            let norm = values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1.0e-4);
            let projection = (0..HEAD_DIM)
                .map(|dim| raw[hidden_base + dim] * values[dim] / norm)
                .sum::<f32>();
            for dim in 0..HEAD_DIM {
                let expected =
                    (raw[hidden_base + dim] - a * projection * values[dim] / norm) * gate;
                common::assert_close(out[hidden_base + dim], expected, 2.0e-4);
            }
        }
    }

    let upstream: Vec<f32> = (0..ROWS * EMBEDDING)
        .map(|index| -0.25 + index as f32 * 0.0625)
        .collect();
    let mut d_out = DeviceBuffer::from_host(&stream, &upstream)?;
    let sentinel = -17.0_f32;
    let mut d_qkv = DeviceBuffer::from_host(&stream, &vec![sentinel; ROWS * QKV_DIM])?;
    let mut d_alphas = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    module.exclusive_self_attention_backward(ExclusiveSelfAttentionBackwardArgs {
        stream: &stream,
        qkv_f16: &qkv_f16,
        raw_out_f16: &raw_f16,
        d_out: &mut d_out,
        d_qkv: &mut d_qkv,
        d_xsa_alphas: &mut d_alphas,
        xsa_alphas: alpha_device,
        row_count: ROWS as u32,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        value_offset: (2 * EMBEDDING) as u32,
        gate_offset: GATE_OFFSET as u32,
        alpha_offset: 0,
        apply_headwise_gate: true,
        kda_value_activation,
    })?;

    let saved_qkv = qkv_f16.to_host_vec(&stream)?;
    let saved_raw = raw_f16.to_host_vec(&stream)?;
    let transformed_grad = d_out.to_host_vec(&stream)?;
    let pre_d_qkv = d_qkv.to_host_vec(&stream)?;
    let d_alphas = d_alphas.to_host_vec(&stream)?;
    let mut expected_alpha_grad = [0.0_f32; HEADS];
    let mut expected_direct_value_grad = vec![0.0_f32; ROWS * QKV_DIM];
    for row in 0..ROWS {
        for head in 0..HEADS {
            let a = alpha_raw[head].tanh();
            let gate_index = row * QKV_DIM + GATE_OFFSET + head;
            let gate = sigmoid(f16_bits_to_f32(saved_qkv[gate_index]));
            let value_base = row * QKV_DIM + 2 * EMBEDDING + head * HEAD_DIM;
            let hidden_base = row * EMBEDDING + head * HEAD_DIM;
            let raw_values: Vec<f32> = (0..HEAD_DIM)
                .map(|dim| f16_bits_to_f32(saved_qkv[value_base + dim]))
                .collect();
            let values: Vec<f32> = raw_values
                .iter()
                .copied()
                .map(|value| xsa_test_value(value, kda_value_activation))
                .collect();
            let norm = values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1.0e-4);
            let unit: Vec<f32> = values.iter().map(|value| value / norm).collect();
            let y: Vec<f32> = (0..HEAD_DIM)
                .map(|dim| f16_bits_to_f32(saved_raw[hidden_base + dim]))
                .collect();
            let projection = (0..HEAD_DIM).map(|dim| y[dim] * unit[dim]).sum::<f32>();
            let exclusive: Vec<f32> = (0..HEAD_DIM)
                .map(|dim| y[dim] - a * projection * unit[dim])
                .collect();
            let dz: Vec<f32> = (0..HEAD_DIM)
                .map(|dim| upstream[hidden_base + dim] * gate)
                .collect();
            let grad_projection = (0..HEAD_DIM).map(|dim| dz[dim] * unit[dim]).sum::<f32>();
            let gate_dot = (0..HEAD_DIM)
                .map(|dim| upstream[hidden_base + dim] * exclusive[dim])
                .sum::<f32>();
            common::assert_close(
                pre_d_qkv[gate_index],
                gate_dot * gate * (1.0 - gate),
                2.0e-4,
            );
            expected_alpha_grad[head] += -(1.0 - a * a) * projection * grad_projection;
            for dim in 0..HEAD_DIM {
                let hidden_index = hidden_base + dim;
                common::assert_close(
                    transformed_grad[hidden_index],
                    dz[dim] - a * grad_projection * unit[dim],
                    2.0e-4,
                );
                let d_value = a
                    * (-grad_projection * y[dim] - projection * dz[dim]
                        + 2.0 * projection * grad_projection * unit[dim])
                    / norm;
                expected_direct_value_grad[value_base + dim] = if kda_value_activation {
                    d_value * silu_test_grad(raw_values[dim])
                } else {
                    d_value
                };
                common::assert_close(
                    pre_d_qkv[value_base + dim],
                    expected_direct_value_grad[value_base + dim],
                    4.0e-4,
                );
            }
        }
        for col in GATE_OFFSET + HEADS..QKV_DIM {
            assert_eq!(pre_d_qkv[row * QKV_DIM + col], 0.0);
        }
    }
    for head in 0..HEADS {
        common::assert_close(d_alphas[head], expected_alpha_grad[head], 3.0e-4);
    }

    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn xsa_test_value(value: f32, kda_value_activation: bool) -> f32 {
    if kda_value_activation {
        value * sigmoid(value)
    } else {
        value
    }
}

fn silu_test_grad(value: f32) -> f32 {
    let sigmoid = sigmoid(value);
    sigmoid * (1.0 + value * (1.0 - sigmoid))
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = 14 - exponent;
        let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let rounded = mantissa + 0x0000_0fff + ((mantissa >> 13) & 1);
    if rounded & 0x0080_0000 != 0 {
        let next_exponent = exponent + 1;
        if next_exponent >= 31 {
            sign | 0x7c00
        } else {
            sign | ((next_exponent as u16) << 10)
        }
    } else {
        sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x03ff) as u32;
    match exponent {
        0 if mantissa == 0 => sign * 0.0,
        0 => sign * (mantissa as f32) * 2.0_f32.powi(-24),
        31 if mantissa == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa as f32 / 1024.0) * 2.0_f32.powi(exponent - 15),
    }
}
