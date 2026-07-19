use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{
    AttentionModule, HeadwiseAttentionGateBackwardArgs, HeadwiseAttentionGateForwardArgs,
};

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

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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
