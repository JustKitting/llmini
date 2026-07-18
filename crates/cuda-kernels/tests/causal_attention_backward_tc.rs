use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{
    AttentionModule, CausalAttentionBackwardTcArgs, CausalAttentionTcArgs,
};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

#[path = "causal_attention_backward_tc/case.rs"]
mod case;
mod common;
#[path = "causal_attention_backward_tc/f16.rs"]
mod f16;
#[path = "causal_attention_backward_tc/reference.rs"]
mod reference;
#[path = "causal_attention_backward_tc/scratch.rs"]
mod scratch;
#[path = "causal_attention_backward_tc/shape.rs"]
mod shape;

use scratch::{TcForwardScratchBuffers, TcScratchBuffers};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn qknorm_backward_matches_scale_finite_difference() -> Result<(), Box<dyn Error>> {
    const SEQ: usize = 16;
    const HEADS: usize = 5;
    const HEAD_DIM: usize = 64;
    const EMBEDDING: usize = HEADS * HEAD_DIM;
    const QKV_DIM: usize = 3 * EMBEDDING;
    const SCALE: f32 = 4.0;
    const EPS: f32 = 0.25;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let attention = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;
    let qkv_values = qknorm_qkv(SEQ, HEADS, HEAD_DIM);
    let d_out_values = qknorm_d_out(SEQ, HEADS, HEAD_DIM);
    let qkv = DeviceBuffer::from_host(&stream, &qkv_values)?;
    let (qkv_f16, qkv_rounded) = f16::saved_f16(&stream, &tc, &qkv_values)?;
    let d_out = DeviceBuffer::from_host(&stream, &d_out_values)?;

    let qk_scale_bytes = DeviceBuffer::from_host(&stream, &[0x02_u8; 8])?;
    let qk_scale_scales = DeviceBuffer::from_host(&stream, &[0x38_u8])?;
    let qk_scale_global_scale = DeviceBuffer::from_host(&stream, &[SCALE])?;
    let qk_scale =
        Nvfp4DeviceTensor::new(&qk_scale_bytes, &qk_scale_scales, &qk_scale_global_scale);
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * EMBEDDING)?;
    let mut attention_out_f16 = DeviceBuffer::<u16>::zeroed(&stream, SEQ * EMBEDDING)?;
    let mut probabilities_f16 = DeviceBuffer::<u16>::zeroed(&stream, HEADS * SEQ * SEQ)?;
    let mut log_sum_exp = DeviceBuffer::<f32>::zeroed(&stream, HEADS * SEQ)?;
    let mut forward_scratch = TcForwardScratchBuffers::new(&stream, HEADS, SEQ, HEAD_DIM)?;
    attention.causal_attention_tc(CausalAttentionTcArgs {
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        qk_scale,
        out: &mut out,
        qkv_f16: None,
        attention_out_f16: Some(&mut attention_out_f16),
        forward_probs_f16: Some(&mut probabilities_f16),
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        log_sum_exp: &mut log_sum_exp,
        scratch: forward_scratch.args(),
        row_count: SEQ as u32,
        seq_len: SEQ as u32,
        batch_size: 1,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        attention_window: SEQ as u32,
    })?;

    let mut softmax_d = DeviceBuffer::<f32>::zeroed(&stream, HEADS * SEQ)?;
    let mut qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * HEADS)?;
    let mut d_qkv = DeviceBuffer::<f32>::zeroed(&stream, SEQ * QKV_DIM)?;
    let mut d_qkv_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 3 * SEQ)?;
    let mut d_qk_scale = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    let mut backward_scratch = TcScratchBuffers::new_for_shape(&stream, HEADS, SEQ, HEAD_DIM)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: true,
        forward_probs_f16: Some(&probabilities_f16),
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv_f16,
        attention_out: &attention_out_f16,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        qk_scale,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut softmax_d,
        qk_norm_max: &mut qk_norm_max,
        d_qkv: &mut d_qkv,
        d_qkv_chunk_amax: &mut d_qkv_chunk_amax,
        d_qk_scale: &mut d_qk_scale,
        scratch: backward_scratch.args(),
        row_count: SEQ as u32,
        seq_len: SEQ as u32,
        batch_size: 1,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        attention_window: SEQ as u32,
        qk_norm_offset: 0,
        backward_mask_seed: 0,
        backward_tile_budget: 0.0,
    })?;

    let plus = qknorm_forward_loss(
        &stream,
        &attention,
        &tc,
        &qkv,
        &d_out_values,
        SCALE + EPS,
        SEQ,
        HEADS,
        HEAD_DIM,
    )?;
    let minus = qknorm_forward_loss(
        &stream,
        &attention,
        &tc,
        &qkv,
        &d_out_values,
        SCALE - EPS,
        SEQ,
        HEADS,
        HEAD_DIM,
    )?;
    let expected_scale_grad = (plus - minus) / (2.0 * EPS);
    let scale_grads = d_qk_scale.to_host_vec(&stream)?;
    let actual_scale_grad = scale_grads[0];
    let tolerance = expected_scale_grad.abs() * 0.08 + 2.0e-3;
    assert!(
        (actual_scale_grad - expected_scale_grad).abs() <= tolerance,
        "qk scale gradient mismatch: actual={actual_scale_grad:.8e} \
         finite_difference={expected_scale_grad:.8e} tolerance={tolerance:.8e}"
    );
    assert!(scale_grads[1..].iter().all(|value| *value == 0.0));

    let gradients = d_qkv.to_host_vec(&stream)?;
    assert_qk_gradients_are_tangent(&qkv_rounded, &gradients, SEQ, HEADS, HEAD_DIM);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn materialized_tc_backward_matches_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let attention = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;
    let case = case::simple_case();

    let (qkv, qkv_ref) = f16::saved_f16(&stream, &tc, &case.qkv)?;
    let (out, out_ref) = f16::saved_f16(&stream, &tc, &case.out)?;
    let (_, d_out_ref) = f16::saved_f16(&stream, &tc, &case.d_out)?;
    let d_out = DeviceBuffer::from_host(&stream, &case.d_out)?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &case.log_sum_exp)?;
    let expected = reference::backward(
        &qkv_ref,
        &out_ref,
        &case.d_out,
        &d_out_ref,
        &case.log_sum_exp,
    );
    let mut tc_softmax_d = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut tc_qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * shape::HEADS)?;
    let mut tc_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut tc_grad_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 3 * shape::TOKEN_COUNT)?;
    let qk_scale_bytes = DeviceBuffer::from_host(&stream, &[0_u8; 8])?;
    let qk_scale_scales = DeviceBuffer::from_host(&stream, &[0x38_u8])?;
    let qk_scale_global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let qk_scale =
        Nvfp4DeviceTensor::new(&qk_scale_bytes, &qk_scale_scales, &qk_scale_global_scale);
    let mut tc_qk_scale_grad = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    let mut scratch = TcScratchBuffers::new(&stream)?;
    let tc_chunk_count = attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: false,
        forward_probs_f16: None,
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        qk_scale,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut tc_softmax_d,
        qk_norm_max: &mut tc_qk_norm_max,
        d_qkv: &mut tc_grad,
        d_qkv_chunk_amax: &mut tc_grad_chunk_amax,
        d_qk_scale: &mut tc_qk_scale_grad,
        scratch: scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
        attention_window: shape::TOKEN_COUNT as u32,
        qk_norm_offset: 0,
        backward_mask_seed: 0,
        backward_tile_budget: 0.0,
    })?;

    let recomputed = tc_grad.to_host_vec(&stream)?;
    common::assert_slice_close(&recomputed, &expected, 1.0e-6);
    let chunk_amax = tc_grad_chunk_amax.to_host_vec(&stream)?;
    assert_eq!(tc_chunk_count as usize, chunk_amax.len());
    for (section, values) in recomputed.chunks(shape::EMBEDDING).enumerate() {
        let expected_amax = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
        assert_eq!(chunk_amax[section].to_bits(), expected_amax.to_bits());
    }

    let mut saved_probs = vec![0_u16; shape::HEADS * shape::TOKEN_COUNT * shape::TOKEN_COUNT];
    for head in 0..shape::HEADS {
        let base = head * shape::TOKEN_COUNT * shape::TOKEN_COUNT;
        saved_probs[base] = 0x3c00;
        saved_probs[base + shape::TOKEN_COUNT] = 0x3800;
        saved_probs[base + shape::TOKEN_COUNT + 1] = 0x3800;
    }
    let mut reuse_softmax_d =
        DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut reuse_qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * shape::HEADS)?;
    let mut reuse_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut reuse_grad_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 3 * shape::TOKEN_COUNT)?;
    let mut reuse_qk_scale_grad = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    let mut reuse_scratch = TcScratchBuffers::new(&stream)?;
    let saved_probs = DeviceBuffer::from_host(&stream, &saved_probs)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: true,
        forward_probs_f16: Some(&saved_probs),
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        qk_scale,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut reuse_softmax_d,
        qk_norm_max: &mut reuse_qk_norm_max,
        d_qkv: &mut reuse_grad,
        d_qkv_chunk_amax: &mut reuse_grad_chunk_amax,
        d_qk_scale: &mut reuse_qk_scale_grad,
        scratch: reuse_scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
        attention_window: shape::TOKEN_COUNT as u32,
        qk_norm_offset: 0,
        backward_mask_seed: 0,
        backward_tile_budget: 0.0,
    })?;
    common::assert_slice_close(&reuse_grad.to_host_vec(&stream)?, &recomputed, 1.0e-6);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn sparse_tile_map_matches_probability_mass_and_seed() -> Result<(), Box<dyn Error>> {
    const SEQ: usize = 1024;
    const HEADS: usize = 1;
    const HEAD_DIM: usize = 64;
    const EMBEDDING: usize = HEADS * HEAD_DIM;
    const QKV_DIM: usize = 3 * EMBEDDING;
    const TILE: usize = 64;
    const TILES: usize = SEQ / TILE;
    const TILE_BUDGET: f32 = 12.0;
    const SEED: u32 = 0x1234_5678;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let attention = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;

    let mut probabilities = vec![0.0_f32; SEQ * SEQ];
    for query in 0..SEQ {
        let probability = 1.0 / (query + 1) as f32;
        for key in 0..=query {
            probabilities[query * SEQ + key] = probability;
        }
    }
    let (probabilities, rounded_probabilities) = f16::saved_f16(&stream, &tc, &probabilities)?;

    let qkv = DeviceBuffer::<u16>::zeroed(&stream, SEQ * QKV_DIM)?;
    let attention_out = DeviceBuffer::<u16>::zeroed(&stream, SEQ * EMBEDDING)?;
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, SEQ * EMBEDDING)?;
    let log_sum_exp = DeviceBuffer::<f32>::zeroed(&stream, SEQ * HEADS)?;
    let mut softmax_d = DeviceBuffer::<f32>::zeroed(&stream, SEQ * HEADS)?;
    let mut qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * HEADS)?;
    let mut d_qkv = DeviceBuffer::<f32>::zeroed(&stream, SEQ * QKV_DIM)?;
    let mut d_qkv_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 3 * SEQ)?;
    let qk_scale_bytes = DeviceBuffer::from_host(&stream, &[0_u8; 8])?;
    let qk_scale_scales = DeviceBuffer::from_host(&stream, &[0x38_u8])?;
    let qk_scale_global_scale = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let qk_scale =
        Nvfp4DeviceTensor::new(&qk_scale_bytes, &qk_scale_scales, &qk_scale_global_scale);
    let mut d_qk_scale = DeviceBuffer::<f32>::zeroed(&stream, 16)?;
    let mut scratch = TcScratchBuffers::new_for_shape(&stream, HEADS, SEQ, HEAD_DIM)?;

    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: true,
        forward_probs_f16: Some(&probabilities),
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &attention_out,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        qk_scale,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut softmax_d,
        qk_norm_max: &mut qk_norm_max,
        d_qkv: &mut d_qkv,
        d_qkv_chunk_amax: &mut d_qkv_chunk_amax,
        d_qk_scale: &mut d_qk_scale,
        scratch: scratch.args(),
        row_count: SEQ as u32,
        seq_len: SEQ as u32,
        batch_size: 1,
        embedding_dim: EMBEDDING as u32,
        qkv_dim: QKV_DIM as u32,
        head_count: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        attention_window: SEQ as u32,
        qk_norm_offset: 0,
        backward_mask_seed: SEED,
        backward_tile_budget: TILE_BUDGET,
    })?;

    let actual_scales = scratch.p().to_host_vec(&stream)?;
    let mut retained_stochastic_tiles = 0;
    let mut dropped_stochastic_tiles = 0;
    for query_tile in 0..TILES {
        for key_tile in 0..TILES {
            let tile_index = query_tile * TILES + key_tile;
            if key_tile > query_tile {
                assert_eq!(actual_scales[tile_index].to_bits(), 0.0_f32.to_bits());
                continue;
            }

            let mut tile_mass = 0.0;
            for query in query_tile * TILE..(query_tile + 1) * TILE {
                for key in key_tile * TILE..(key_tile + 1) * TILE {
                    tile_mass += rounded_probabilities[query * SEQ + key];
                }
            }
            let q = (TILE_BUDGET * tile_mass / TILE as f32).min(1.0);
            let random = (hash_u32(SEED ^ (tile_index as u32).wrapping_mul(0x9e37_79b9)) >> 8)
                as f32
                / 16_777_216.0;
            let expected = if q > 0.0 && random < q { 1.0 / q } else { 0.0 };
            let actual = actual_scales[tile_index];
            let tolerance = expected.abs() * 1.0e-4 + 1.0e-6;
            assert!(
                (actual - expected).abs() <= tolerance,
                "tile ({query_tile}, {key_tile}) mismatch: actual={actual}, expected={expected}, \
                 q={q}, random={random}"
            );
            if q < 1.0 {
                if expected == 0.0 {
                    dropped_stochastic_tiles += 1;
                } else {
                    retained_stochastic_tiles += 1;
                }
            }
        }
    }
    assert!(retained_stochastic_tiles > 0);
    assert!(dropped_stochastic_tiles > 0);
    assert!(
        d_qkv
            .to_host_vec(&stream)?
            .iter()
            .all(|value| *value == 0.0)
    );
    Ok(())
}

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn qknorm_qkv(seq_len: usize, head_count: usize, head_dim: usize) -> Vec<f32> {
    let embedding = head_count * head_dim;
    let qkv_dim = 3 * embedding;
    let mut qkv = vec![0.0_f32; seq_len * qkv_dim];
    for token in 0..seq_len {
        for col in 0..embedding {
            let q = ((token * 5 + col * 3) % 17) as i32 - 8;
            let k = ((token * 7 + col * 5 + 3) % 19) as i32 - 9;
            let v = ((token * 11 + col * 2 + 1) % 23) as i32 - 11;
            let row = token * qkv_dim;
            qkv[row + col] = q as f32 * 0.0625;
            qkv[row + embedding + col] = k as f32 * 0.0625;
            qkv[row + 2 * embedding + col] = v as f32 * 0.03125;
        }
    }
    qkv
}

fn qknorm_d_out(seq_len: usize, head_count: usize, head_dim: usize) -> Vec<f32> {
    let embedding = head_count * head_dim;
    let mut values = vec![0.0_f32; seq_len * embedding];
    for token in 0..seq_len {
        for col in 0..embedding {
            let value = ((token * 3 + col * 7 + 2) % 13) as i32 - 6;
            values[token * embedding + col] = value as f32 * 0.03125;
        }
    }
    values
}

#[expect(
    clippy::too_many_arguments,
    reason = "the finite-difference helper mirrors the CUDA call"
)]
fn qknorm_forward_loss(
    stream: &cuda_core::CudaStream,
    attention: &AttentionModule,
    tc: &F16TcMatmulModule,
    qkv: &DeviceBuffer<f32>,
    d_out: &[f32],
    scale: f32,
    seq_len: usize,
    head_count: usize,
    head_dim: usize,
) -> Result<f32, Box<dyn Error>> {
    let embedding = head_count * head_dim;
    let qk_scale_bytes = DeviceBuffer::from_host(stream, &[0x02_u8; 8])?;
    let qk_scale_scales = DeviceBuffer::from_host(stream, &[0x38_u8])?;
    let qk_scale_global_scale = DeviceBuffer::from_host(stream, &[scale])?;
    let qk_scale =
        Nvfp4DeviceTensor::new(&qk_scale_bytes, &qk_scale_scales, &qk_scale_global_scale);
    let mut out = DeviceBuffer::<f32>::zeroed(stream, seq_len * embedding)?;
    let mut log_sum_exp = DeviceBuffer::<f32>::zeroed(stream, head_count * seq_len)?;
    let mut scratch = TcForwardScratchBuffers::new(stream, head_count, seq_len, head_dim)?;
    attention.causal_attention_tc(CausalAttentionTcArgs {
        stream,
        tc_module: tc,
        qkv,
        qk_scale,
        out: &mut out,
        qkv_f16: None,
        attention_out_f16: None,
        forward_probs_f16: None,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        log_sum_exp: &mut log_sum_exp,
        scratch: scratch.args(),
        row_count: seq_len as u32,
        seq_len: seq_len as u32,
        batch_size: 1,
        embedding_dim: embedding as u32,
        qkv_dim: (3 * embedding) as u32,
        head_count: head_count as u32,
        head_dim: head_dim as u32,
        attention_window: seq_len as u32,
    })?;
    Ok(out
        .to_host_vec(stream)?
        .iter()
        .zip(d_out)
        .map(|(out, grad)| out * grad)
        .sum())
}

fn assert_qk_gradients_are_tangent(
    qkv_rotated: &[f32],
    gradients_raw: &[f32],
    seq_len: usize,
    head_count: usize,
    head_dim: usize,
) {
    let embedding = head_count * head_dim;
    let qkv_dim = 3 * embedding;
    for token in 0..seq_len {
        for section in 0..2 {
            for head in 0..head_count {
                let offset = section * embedding + head * head_dim;
                let mut dot = 0.0_f32;
                let mut value_sumsq = 0.0_f32;
                let mut grad_sumsq = 0.0_f32;
                for dim in (0..head_dim).step_by(2) {
                    let angle = token as f32 * (-9.210_340_5 * dim as f32 / head_dim as f32).exp();
                    let (sin, cos) = angle.sin_cos();
                    let base = token * qkv_dim + offset + dim;
                    let grad_even = gradients_raw[base];
                    let grad_odd = gradients_raw[base + 1];
                    let rotated_grad_even = grad_even * cos - grad_odd * sin;
                    let rotated_grad_odd = grad_odd * cos + grad_even * sin;
                    let value_even = qkv_rotated[base];
                    let value_odd = qkv_rotated[base + 1];
                    dot += value_even * rotated_grad_even + value_odd * rotated_grad_odd;
                    value_sumsq += value_even * value_even + value_odd * value_odd;
                    grad_sumsq +=
                        rotated_grad_even * rotated_grad_even + rotated_grad_odd * rotated_grad_odd;
                }
                let scale = (value_sumsq * grad_sumsq).sqrt();
                let tolerance = scale * 2.0e-3 + 1.0e-5;
                assert!(
                    dot.abs() <= tolerance,
                    "normalization VJP is not tangent at token={token} head={head} \
                     section={section}: dot={dot:.8e} tolerance={tolerance:.8e}"
                );
            }
        }
    }
}
