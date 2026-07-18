use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardTcArgs};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

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

use scratch::TcScratchBuffers;

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
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut tc_softmax_d,
        qk_norm_max: &mut tc_qk_norm_max,
        d_qkv: &mut tc_grad,
        d_qkv_chunk_amax: &mut tc_grad_chunk_amax,
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
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut reuse_softmax_d,
        qk_norm_max: &mut reuse_qk_norm_max,
        d_qkv: &mut reuse_grad,
        d_qkv_chunk_amax: &mut reuse_grad_chunk_amax,
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
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut softmax_d,
        qk_norm_max: &mut qk_norm_max,
        d_qkv: &mut d_qkv,
        d_qkv_chunk_amax: &mut d_qkv_chunk_amax,
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
