use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::{AttentionModule, CProjArgs, QkvProjectionArgs};
use rust_kernels_cuda::f16_tc_matmul::{F16ConvertArgs, F16TcMatmulModule};
use rust_kernels_cuda::lm_head::{LmHeadArgs, LmHeadModule};
use rust_kernels_cuda::mlp::{
    MlpBlockTopKRouteArgs, MlpDownResidualArgs, MlpModule, MlpUpRelu2Args, Relu2BackwardF16Args,
    relu2_backward_amax_chunks,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    kernels::{
        TILE_K, TILE_M, TILE_N, polar::TILE_N as POLAR_TILE_N, tma_nvfp4_output_amax_chunks,
    },
    launcher::{Nvfp4GemmModule, Nvfp4GemmRoute},
    pad::{TmaMatrixPadModule, U4RowPadArgs},
    scale_layout::{
        pack_sm120_scale_plane_compact_padded, sm120_scale_packed_len, sm120_scale_padded_mn_extent,
    },
    scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

mod common;

use common::nvfp4::{one_scales, set_e2m1_one};

const ROWS: usize = 128;
const K: usize = 128;
const N: usize = 160;
const TOLERANCE: f32 = 1.0e-5;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn mlp_block_topk_route_uses_only_prior_sequence_context_and_transposes_exact_tiles()
-> Result<(), Box<dyn Error>> {
    const TILE: usize = 128;
    const TOKEN_TILES: usize = 64;
    const FEATURE_TILES: usize = 64;
    const ACTIVE_FEATURE_TILES: usize = 48;
    const TOKENS: usize = TILE * TOKEN_TILES;
    const FEATURES: usize = TILE * FEATURE_TILES;
    const ELEMENTS: usize = TOKENS * FEATURES;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = MlpModule::from_module(ptx)?;
    let mut bytes = vec![0_u8; ELEMENTS / 2];
    for (row_index, row) in bytes.chunks_exact_mut(FEATURES / 2).enumerate() {
        let token_tile = row_index / TILE;
        for feature_tile in 0..FEATURE_TILES {
            let e2m1 = ((feature_tile * 5 + token_tile * 3) % 8) as u8;
            let pair = e2m1 | (e2m1 << 4);
            let byte_base = feature_tile * TILE / 2;
            row[byte_base..byte_base + TILE / 2].fill(pair);
        }
    }
    let bytes = DeviceBuffer::from_host(&stream, &bytes)?;
    let logical_scales: Vec<u8> = (0..ELEMENTS / 16)
        .map(|i| {
            let row = i / (FEATURES / 16);
            let group = i % (FEATURES / 16);
            0x30 + (((group / 8 + row + group % 8) % 5) * 8) as u8
        })
        .collect();
    let scales = DeviceBuffer::from_host(
        &stream,
        &common::encode_csf_scales(&logical_scales, TOKENS, FEATURES),
    )?;
    let globals: Vec<f32> = (0..TOKENS).map(|r| 1.0 + (r % 7) as f32 * 0.25).collect();
    let global_scales = DeviceBuffer::from_host(&stream, &globals)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_TILES * FEATURE_TILES)?;
    let mut masks = DeviceBuffer::<u64>::zeroed(&stream, TOKEN_TILES + FEATURE_TILES)?;

    module.block_topk_route(MlpBlockTopKRouteArgs {
        stream: &stream,
        activation: Nvfp4RowwiseDeviceTensor::new(&bytes, &scales, &global_scales),
        scores: &mut scores,
        masks: &mut masks,
        token_count: TOKENS as u32,
        seq_len: 2048,
        feature_count: FEATURES as u32,
    })?;

    let scores = scores.to_host_vec(&stream)?;
    let masks = masks.to_host_vec(&stream)?;
    let expected_scores: Vec<f32> = (0..TOKEN_TILES * FEATURE_TILES)
        .map(|i| {
            let t = i / FEATURE_TILES;
            let f = i % FEATURE_TILES;
            let sequence_start = t % (2048 / TILE) == 0;
            let source_tile = if sequence_start { t } else { t - 1 };
            let source_rows = if sequence_start { 1 } else { TILE };
            let payload =
                [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(f * 5 + source_tile * 3) % 8];
            let mut expected = 0.0f32;
            for row in source_tile * TILE..source_tile * TILE + source_rows {
                for group in 0..8 {
                    let scale = [0.5f32, 1.0, 2.0, 4.0, 8.0][(f + row + group) % 5];
                    expected += 16.0 * payload * scale * globals[row];
                }
            }
            expected
        })
        .collect();
    assert_eq!(
        scores, expected_scores,
        "compressed scales changed route scores"
    );
    let mut expected_masks = vec![0_u64; TOKEN_TILES + FEATURE_TILES];
    for token_tile in 0..TOKEN_TILES {
        let score_base = token_tile * FEATURE_TILES;
        let mut ranked_features: Vec<_> = (0..FEATURE_TILES).collect();
        ranked_features.sort_unstable_by(|left, right| {
            expected_scores[score_base + *right]
                .total_cmp(&expected_scores[score_base + *left])
                .then_with(|| left.cmp(right))
        });
        for feature_tile in ranked_features[..ACTIVE_FEATURE_TILES].iter().copied() {
            expected_masks[token_tile] |= 1_u64 << feature_tile;
            expected_masks[TOKEN_TILES + feature_tile] |= 1_u64 << token_tile;
        }
    }
    assert_eq!(masks, expected_masks);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn mlp_block_topk_route_batch8_preserves_upper_half_and_causality() -> Result<(), Box<dyn Error>> {
    const TILE: usize = 128;
    const TOKEN_TILES: usize = 128;
    const FEATURE_TILES: usize = 64;
    const ACTIVE_FEATURE_TILES: usize = 48;
    const TOKENS: usize = TILE * TOKEN_TILES;
    const FEATURES: usize = TILE * FEATURE_TILES;
    const ELEMENTS: usize = TOKENS * FEATURES;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = MlpModule::from_module(ptx)?;
    let mut bytes = vec![0_u8; ELEMENTS / 2];
    for (row_index, row) in bytes.chunks_exact_mut(FEATURES / 2).enumerate() {
        let token_tile = row_index / TILE;
        for feature_tile in 0..FEATURE_TILES {
            let e2m1 = ((feature_tile * 5 + token_tile * 3) % 8) as u8;
            let pair = e2m1 | (e2m1 << 4);
            let byte_base = feature_tile * TILE / 2;
            row[byte_base..byte_base + TILE / 2].fill(pair);
        }
    }
    let bytes = DeviceBuffer::from_host(&stream, &bytes)?;
    let logical_scales: Vec<u8> = (0..ELEMENTS / 16)
        .map(|i| {
            let row = i / (FEATURES / 16);
            let group = i % (FEATURES / 16);
            0x30 + (((group / 8 + row + group % 8) % 5) * 8) as u8
        })
        .collect();
    let scales = DeviceBuffer::from_host(
        &stream,
        &common::encode_csf_scales(&logical_scales, TOKENS, FEATURES),
    )?;
    let globals: Vec<f32> = (0..TOKENS).map(|r| 1.0 + (r % 7) as f32 * 0.25).collect();
    let global_scales = DeviceBuffer::from_host(&stream, &globals)?;
    let mut scores = DeviceBuffer::<f32>::zeroed(&stream, TOKEN_TILES * FEATURE_TILES)?;
    let mut masks = DeviceBuffer::<u64>::zeroed(&stream, TOKEN_TILES + FEATURE_TILES * 2)?;

    module.block_topk_route(MlpBlockTopKRouteArgs {
        stream: &stream,
        activation: Nvfp4RowwiseDeviceTensor::new(&bytes, &scales, &global_scales),
        scores: &mut scores,
        masks: &mut masks,
        token_count: TOKENS as u32,
        seq_len: 2048,
        feature_count: FEATURES as u32,
    })?;

    let scores = scores.to_host_vec(&stream)?;
    let masks = masks.to_host_vec(&stream)?;
    let expected_scores: Vec<f32> = (0..TOKEN_TILES * FEATURE_TILES)
        .map(|i| {
            let t = i / FEATURE_TILES;
            let f = i % FEATURE_TILES;
            let sequence_start = t % (2048 / TILE) == 0;
            let source_tile = if sequence_start { t } else { t - 1 };
            let source_rows = if sequence_start { 1 } else { TILE };
            let payload =
                [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(f * 5 + source_tile * 3) % 8];
            let mut expected = 0.0f32;
            for row in source_tile * TILE..source_tile * TILE + source_rows {
                for group in 0..8 {
                    let scale = [0.5f32, 1.0, 2.0, 4.0, 8.0][(f + row + group) % 5];
                    expected += 16.0 * payload * scale * globals[row];
                }
            }
            expected
        })
        .collect();
    assert_eq!(
        scores, expected_scores,
        "compressed scales changed route scores"
    );
    let mut expected_masks = vec![0_u64; TOKEN_TILES + FEATURE_TILES * 2];
    for token_tile in 0..TOKEN_TILES {
        let score_base = token_tile * FEATURE_TILES;
        let mut ranked_features: Vec<_> = (0..FEATURE_TILES).collect();
        ranked_features.sort_unstable_by(|left, right| {
            expected_scores[score_base + *right]
                .total_cmp(&expected_scores[score_base + *left])
                .then_with(|| left.cmp(right))
        });
        for feature_tile in ranked_features[..ACTIVE_FEATURE_TILES].iter().copied() {
            expected_masks[token_tile] |= 1_u64 << feature_tile;
            expected_masks[TOKEN_TILES + feature_tile * 2 + token_tile / 64] |=
                1_u64 << (token_tile % 64);
        }
    }
    assert_eq!(masks, expected_masks);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_feature_routes_use_transposed_popcount_and_zero_empty_tiles() -> Result<(), Box<dyn Error>> {
    let masks = [0b01_u64, 0b01, 0b11, 0];

    let feature_n = Fixture::new(128, 256, 256)?;
    let masks_n = DeviceBuffer::from_host(&feature_n.stream, &masks)?;
    let mut dense_n = DeviceBuffer::<f32>::zeroed(&feature_n.stream, 128 * 256)?;
    let mut routed_n = DeviceBuffer::<f32>::zeroed(&feature_n.stream, 128 * 256)?;
    feature_n.tma_rowwise_exact(&mut dense_n, None)?;
    feature_n.tma_rowwise_exact(
        &mut routed_n,
        Some(Nvfp4GemmRoute::feature_n_token_k(&masks_n, 2, 2, 1)),
    )?;
    let dense_n = dense_n.to_host_vec(&feature_n.stream)?;
    let routed_n = routed_n.to_host_vec(&feature_n.stream)?;
    for row in 0..128 {
        let base = row * 256;
        common::assert_slice_close(
            &routed_n[base..base + 128],
            &dense_n[base..base + 128],
            TOLERANCE,
        );
        assert!(
            routed_n[base + 128..base + 256]
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    let feature_m = Fixture::new(256, 256, 128)?;
    let masks_m = DeviceBuffer::from_host(&feature_m.stream, &masks)?;
    let mut dense_m = DeviceBuffer::<f32>::zeroed(&feature_m.stream, 256 * 128)?;
    let mut routed_m = DeviceBuffer::<f32>::zeroed(&feature_m.stream, 256 * 128)?;
    feature_m.tma_rowwise_exact(&mut dense_m, None)?;
    feature_m.tma_rowwise_exact(
        &mut routed_m,
        Some(Nvfp4GemmRoute::feature_m_token_k(&masks_m, 2, 2, 1)),
    )?;
    let dense_m = dense_m.to_host_vec(&feature_m.stream)?;
    let routed_m = routed_m.to_host_vec(&feature_m.stream)?;
    common::assert_slice_close(&routed_m[..128 * 128], &dense_m[..128 * 128], TOLERANCE);
    assert!(routed_m[128 * 128..].iter().all(|value| *value == 0.0));
    Ok(())
}

// Only selected K tiles contain nonzero values. Dense CUDA output is therefore
// an independent reference for each routed backward orientation, including the
// 63/64 word boundary and last tile of a physical batch of eight sequences.
#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_feature_routes_consume_batch8_upper_token_word() -> Result<(), Box<dyn Error>> {
    const K: usize = 128 * 128;
    for selected in [63_usize, 64, 127] {
        for feature_on_m in [false, true] {
            let (rows, n) = if feature_on_m { (256, 128) } else { (128, 256) };
            let mut fixture = Fixture::new(rows, K, n)?;
            let sparse = |count: usize| {
                let mut bytes = vec![0_u8; count * K / 2];
                for row in bytes.chunks_exact_mut(K / 2) {
                    row[selected * 64..selected * 64 + 64].fill(0x22);
                }
                bytes
            };
            fixture.input_bytes = DeviceBuffer::from_host(&fixture.stream, &sparse(rows))?;
            fixture.weight_bytes = DeviceBuffer::from_host(&fixture.stream, &sparse(n))?;
            let mut masks = vec![0_u64; 128 + 2 * 2];
            masks[selected] = 1;
            masks[128 + selected / 64] = 1_u64 << (selected % 64);
            let masks = DeviceBuffer::from_host(&fixture.stream, &masks)?;
            let mut dense = DeviceBuffer::<f32>::zeroed(&fixture.stream, rows * n)?;
            let mut routed = DeviceBuffer::<f32>::zeroed(&fixture.stream, rows * n)?;
            fixture.tma_rowwise_exact(&mut dense, None)?;
            let route = if feature_on_m {
                Nvfp4GemmRoute::feature_m_token_k(&masks, 128, 2, 1)
            } else {
                Nvfp4GemmRoute::feature_n_token_k(&masks, 128, 2, 1)
            };
            fixture.tma_rowwise_exact(&mut routed, Some(route))?;
            let dense = dense.to_host_vec(&fixture.stream)?;
            let routed = routed.to_host_vec(&fixture.stream)?;
            assert!(dense.iter().all(|v| v.is_finite() && *v > 0.0));
            for row in 0..rows {
                for col in 0..n {
                    let active = if feature_on_m { row < 128 } else { col < 128 };
                    let index = row * n + col;
                    let expected = if active { dense[index] } else { 0.0 };
                    assert!(
                        (routed[index] - expected).abs() <= TOLERANCE,
                        "tile={selected} feature_on_m={feature_on_m} row={row} col={col}: {} != {expected}",
                        routed[index]
                    );
                }
            }
        }
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_scale_pack_u32_matches_cpu_layout() -> Result<(), Box<dyn Error>> {
    const MN: usize = 160;
    const K_DIM: usize = 256;
    let logical = (0..MN * K_DIM / 16)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect::<Vec<_>>();
    let expected = pack_sm120_scale_plane_compact_padded(&logical, MN, K_DIM);
    let (_, stream, ptx) = common::cuda_test_context()?;
    let scale_pack = Sm120ScalePackModule::from_module(ptx)?;
    let logical = DeviceBuffer::from_host(&stream, &logical)?;
    let mut packed = DeviceBuffer::<u8>::zeroed(&stream, expected.len())?;
    scale_pack.pack(&stream, &logical, &mut packed, MN as u32, K_DIM as u32)?;
    assert_eq!(packed.to_host_vec(&stream)?, expected);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_scale_pack_pair_matches_cpu_layout() -> Result<(), Box<dyn Error>> {
    const MN_0: usize = 160;
    const MN_1: usize = 257;
    let (_, stream, ptx) = common::cuda_test_context()?;
    let scale_pack = Sm120ScalePackModule::from_module(ptx)?;

    for k_dim in [192usize, 256] {
        let logical_0 = (0..MN_0 * k_dim / 16)
            .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
            .collect::<Vec<_>>();
        let logical_1 = (0..MN_1 * k_dim / 16)
            .map(|index| (index.wrapping_mul(53).wrapping_add(29) & 0xff) as u8)
            .collect::<Vec<_>>();
        let expected_0 = pack_sm120_scale_plane_compact_padded(&logical_0, MN_0, k_dim);
        let expected_1 = pack_sm120_scale_plane_compact_padded(&logical_1, MN_1, k_dim);
        let logical_0 = DeviceBuffer::from_host(&stream, &logical_0)?;
        let logical_1 = DeviceBuffer::from_host(&stream, &logical_1)?;
        let mut packed_0 = DeviceBuffer::<u8>::zeroed(&stream, expected_0.len())?;
        let mut packed_1 = DeviceBuffer::<u8>::zeroed(&stream, expected_1.len())?;

        scale_pack.pack_pair(
            &stream,
            &logical_0,
            &mut packed_0,
            MN_0 as u32,
            &logical_1,
            &mut packed_1,
            MN_1 as u32,
            k_dim as u32,
        )?;
        assert_eq!(packed_0.to_host_vec(&stream)?, expected_0);
        assert_eq!(packed_1.to_host_vec(&stream)?, expected_1);
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_scale_pack_quad_matches_cpu_layout() -> Result<(), Box<dyn Error>> {
    const MN_0: usize = 160;
    const MN_1: usize = 257;
    const MN_2: usize = 129;
    const MN_3: usize = 384;
    const K_DIM_01: usize = 192;
    const K_DIM_23: usize = 256;

    fn logical_plane(mn: usize, k_dim: usize, multiplier: usize, addend: usize) -> Vec<u8> {
        (0..mn * k_dim / 16)
            .map(|index| (index.wrapping_mul(multiplier).wrapping_add(addend) & 0xff) as u8)
            .collect()
    }

    let logical_0 = logical_plane(MN_0, K_DIM_01, 37, 11);
    let logical_1 = logical_plane(MN_1, K_DIM_01, 53, 29);
    let logical_2 = logical_plane(MN_2, K_DIM_23, 71, 17);
    let logical_3 = logical_plane(MN_3, K_DIM_23, 89, 43);
    let expected_0 = pack_sm120_scale_plane_compact_padded(&logical_0, MN_0, K_DIM_01);
    let expected_1 = pack_sm120_scale_plane_compact_padded(&logical_1, MN_1, K_DIM_01);
    let expected_2 = pack_sm120_scale_plane_compact_padded(&logical_2, MN_2, K_DIM_23);
    let expected_3 = pack_sm120_scale_plane_compact_padded(&logical_3, MN_3, K_DIM_23);

    let (_, stream, ptx) = common::cuda_test_context()?;
    let scale_pack = Sm120ScalePackModule::from_module(ptx)?;
    let logical_0 = DeviceBuffer::from_host(&stream, &logical_0)?;
    let logical_1 = DeviceBuffer::from_host(&stream, &logical_1)?;
    let logical_2 = DeviceBuffer::from_host(&stream, &logical_2)?;
    let logical_3 = DeviceBuffer::from_host(&stream, &logical_3)?;
    let mut packed_0 = DeviceBuffer::<u8>::zeroed(&stream, expected_0.len())?;
    let mut packed_1 = DeviceBuffer::<u8>::zeroed(&stream, expected_1.len())?;
    let mut packed_2 = DeviceBuffer::<u8>::zeroed(&stream, expected_2.len())?;
    let mut packed_3 = DeviceBuffer::<u8>::zeroed(&stream, expected_3.len())?;

    scale_pack.pack_quad(
        &stream,
        &logical_0,
        &mut packed_0,
        MN_0 as u32,
        &logical_1,
        &mut packed_1,
        MN_1 as u32,
        K_DIM_01 as u32,
        &logical_2,
        &mut packed_2,
        MN_2 as u32,
        &logical_3,
        &mut packed_3,
        MN_3 as u32,
        K_DIM_23 as u32,
    )?;
    assert_eq!(packed_0.to_host_vec(&stream)?, expected_0);
    assert_eq!(packed_1.to_host_vec(&stream)?, expected_1);
    assert_eq!(packed_2.to_host_vec(&stream)?, expected_2);
    assert_eq!(packed_3.to_host_vec(&stream)?, expected_3);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_raw_padded_output_matches_old_lm_head_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, N)?;
    let mut old_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;
    let mut tma_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;

    fixture.old_raw(&mut old_out)?;
    fixture.tma_raw(&mut tma_out)?;

    common::assert_slice_close(
        &tma_out.to_host_vec(&fixture.stream)?,
        &old_out.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_output_amax_matches_stored_output() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut fused = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut fused_amax = DeviceBuffer::<f32>::zeroed(&fixture.stream, 1)?;

    fixture.tma_scalar_with_output_amax(&mut plain, &mut fused, &mut fused_amax)?;

    let plain = plain.to_host_vec(&fixture.stream)?;
    let fused = fused.to_host_vec(&fixture.stream)?;
    assert_eq!(fused, plain, "amax epilogue changed the stored GEMM output");

    let output_amax = fused.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let fused_amax = fused_amax.to_host_vec(&fixture.stream)?[0];
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_relu2_backward_f16_amax_matches_standalone_epilogue() -> Result<(), Box<dyn Error>> {
    const DIM: usize = 128;
    const LEN: usize = ROWS * DIM;
    let fixture = Fixture::new(ROWS, K, DIM)?;
    let pre_f32 = DeviceBuffer::from_host(&fixture.stream, &pre_activation_values(LEN))?;
    let mut pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, LEN)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut standalone = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let mut fused = DeviceBuffer::<f32>::zeroed(&fixture.stream, LEN)?;
    let standalone_chunk_count = relu2_backward_amax_chunks(LEN as u32);
    // The GEMM and standalone reduction can partition the same output differently.
    let fused_chunk_count = tma_nvfp4_output_amax_chunks(ROWS as u32, DIM as u32);
    let mut standalone_chunks =
        DeviceBuffer::<f32>::zeroed(&fixture.stream, standalone_chunk_count as usize)?;
    let mut fused_chunks =
        DeviceBuffer::<f32>::zeroed(&fixture.stream, fused_chunk_count as usize)?;

    fixture.f16.fp32_to_f16(F16ConvertArgs {
        stream: &fixture.stream,
        src: &pre_f32,
        dst: &mut pre_f16,
        element_count: LEN as u32,
    })?;
    let launched_chunks = fixture.tma_relu2_backward_f16_with_output_amax(
        &pre_f16,
        &mut plain,
        &mut fused,
        &mut fused_chunks,
    )?;
    assert_eq!(launched_chunks, fused_chunk_count);
    fixture.mlp.relu2_backward_f16(Relu2BackwardF16Args {
        stream: &fixture.stream,
        pre_activation: &pre_f16,
        d_out: &plain,
        d_pre_activation: &mut standalone,
        d_pre_activation_chunk_amax: &mut standalone_chunks,
        len: LEN as u32,
    })?;

    let standalone = standalone.to_host_vec(&fixture.stream)?;
    let fused = fused.to_host_vec(&fixture.stream)?;
    assert_eq!(
        fused, standalone,
        "fused ReLU2 backward epilogue changed the stored gradient"
    );
    let output_amax = fused.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let fused_amax = fused_chunks
        .to_host_vec(&fixture.stream)?
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_symmetric_output_and_amax_match_full_self_product() -> Result<(), Box<dyn Error>> {
    const DIM: usize = 256;
    let fixture = Fixture::new(DIM, K, DIM)?;
    let mut plain = DeviceBuffer::<f32>::zeroed(&fixture.stream, DIM * DIM)?;
    let mut symmetric = DeviceBuffer::<f32>::zeroed(&fixture.stream, DIM * DIM)?;
    let mut fused_amax = DeviceBuffer::<f32>::zeroed(&fixture.stream, 1)?;

    fixture.tma_self_symmetric_with_output_amax(&mut plain, &mut symmetric, &mut fused_amax)?;

    let plain = plain.to_host_vec(&fixture.stream)?;
    let symmetric = symmetric.to_host_vec(&fixture.stream)?;
    assert_eq!(
        symmetric, plain,
        "triangular self-product changed the stored Gram matrix"
    );

    let output_amax = symmetric
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let fused_amax = fused_amax.to_host_vec(&fixture.stream)?[0];
    assert_eq!(fused_amax, output_amax);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_affine_padded_output_matches_old_qkv_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, N)?;
    let mut old_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;
    let mut tma_out = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * N)?;

    fixture.attention.qkv_projection(QkvProjectionArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        out: &mut old_out,
        token_count: ROWS as u32,
        input_dim: K as u32,
        output_dim: N as u32,
    })?;
    fixture.tma_affine(&mut tma_out)?;

    common::assert_slice_close(
        &tma_out.to_host_vec(&fixture.stream)?,
        &old_out.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_affine_columns_replay_preserves_full_stride_and_other_columns() -> Result<(), Box<dyn Error>>
{
    let tile = TILE_N as usize;
    let n = 3 * tile + 32;
    let fixture = Fixture::new(ROWS, 3 * K, n)?;
    let mut full = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * n)?;
    let poison = -1234.25_f32;
    let mut partial = DeviceBuffer::from_host(&fixture.stream, &vec![poison; ROWS * n])?;
    fixture.tma_affine(&mut full)?;
    fixture.tma_affine_columns(&mut partial, Some(TILE_N..3 * TILE_N))?;
    let full = full.to_host_vec(&fixture.stream)?;
    let partial = partial.to_host_vec(&fixture.stream)?;
    for row in 0..ROWS {
        for column in 0..n {
            let index = row * n + column;
            let expected = if (tile..3 * tile).contains(&column) {
                full[index]
            } else {
                poison
            };
            assert_eq!(
                partial[index].to_bits(),
                expected.to_bits(),
                "row {row}, column {column}"
            );
        }
    }
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_residual_matches_old_projection_residual_add() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let residual = residual_values(ROWS * 128);
    let mut old_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;
    let mut tma_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;

    fixture.attention.c_proj(CProjArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        residual: &mut old_residual,
        token_count: ROWS as u32,
        embedding_dim: 128,
    })?;
    fixture.tma_residual(&mut tma_residual)?;

    common::assert_slice_close(
        &tma_residual.to_host_vec(&fixture.stream)?,
        &old_residual.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_relu2_matches_old_mlp_up_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, K, 128)?;
    let mut old_pre = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut old_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_pre = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut compact_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut reference_pre_f16 = DeviceBuffer::<u16>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut tma_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;
    let mut compact_act = DeviceBuffer::<f32>::zeroed(&fixture.stream, ROWS * 128)?;

    fixture.mlp.up_relu2(MlpUpRelu2Args {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        pre_activation: &mut old_pre,
        out: &mut old_act,
        token_count: ROWS as u32,
        input_dim: K as u32,
        output_dim: 128,
    })?;
    fixture.tma_relu2(&mut tma_pre, Some(&mut tma_pre_f16), &mut tma_act)?;
    fixture.tma_relu2_compact(Some(&mut compact_pre_f16), &mut compact_act)?;
    fixture.f16.fp32_to_f16(F16ConvertArgs {
        stream: &fixture.stream,
        src: &tma_pre,
        dst: &mut reference_pre_f16,
        element_count: (ROWS * 128) as u32,
    })?;

    common::assert_slice_close(
        &tma_pre.to_host_vec(&fixture.stream)?,
        &old_pre.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    common::assert_slice_close(
        &tma_act.to_host_vec(&fixture.stream)?,
        &old_act.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    assert_eq!(
        compact_act.to_host_vec(&fixture.stream)?,
        tma_act.to_host_vec(&fixture.stream)?,
        "compact ReLU2 epilogue changed the activation"
    );
    assert_eq!(
        tma_pre_f16.to_host_vec(&fixture.stream)?,
        reference_pre_f16.to_host_vec(&fixture.stream)?,
        "fused ReLU2 tape must match the standalone FP32-to-FP16 conversion"
    );
    assert_eq!(
        compact_pre_f16.to_host_vec(&fixture.stream)?,
        reference_pre_f16.to_host_vec(&fixture.stream)?,
        "compact ReLU2 tape must match the standalone FP32-to-FP16 conversion"
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_mlp_down_residual_matches_old_projection() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(ROWS, 256, 128)?;
    let residual = residual_values(ROWS * 128);
    let mut old_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;
    let mut tma_residual = DeviceBuffer::from_host(&fixture.stream, &residual)?;

    fixture.mlp.down_residual(MlpDownResidualArgs {
        stream: &fixture.stream,
        input: fixture.input(),
        weight: fixture.weight_mma(),
        bias: fixture.bias_device(),
        residual: &mut old_residual,
        token_count: ROWS as u32,
        input_dim: 256,
        output_dim: 128,
    })?;
    fixture.tma_residual(&mut tma_residual)?;

    common::assert_slice_close(
        &tma_residual.to_host_vec(&fixture.stream)?,
        &old_residual.to_host_vec(&fixture.stream)?,
        TOLERANCE,
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tma_branch_output_overwrites_poisoned_dense_and_routed_outputs() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new(
        (2 * TILE_M) as usize,
        (4 * TILE_K) as usize,
        (2 * TILE_N) as usize,
    )?;
    let masks = DeviceBuffer::from_host(&fixture.stream, &[0b0111_u64, 0b1101, 0, 0, 0, 0])?;
    let route = Nvfp4GemmRoute::token_m_feature_k(&masks, 2, 4, 3);
    let len = fixture.rows * fixture.n;
    for route in [None, Some(route)] {
        let mut zero_residual = DeviceBuffer::<f32>::zeroed(&fixture.stream, len)?;
        let mut poisoned = DeviceBuffer::from_host(&fixture.stream, &vec![f32::NAN; len])?;
        fixture.tma_projection(&mut zero_residual, route, true)?;
        fixture.tma_projection(&mut poisoned, route, false)?;
        let expected = zero_residual.to_host_vec(&fixture.stream)?;
        let actual = poisoned.to_host_vec(&fixture.stream)?;
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            assert!(
                expected.is_finite(),
                "reference output {index} is nonfinite"
            );
            assert_eq!(actual.to_bits(), expected.to_bits(), "output index {index}");
        }
    }
    Ok(())
}

struct Fixture {
    stream: std::sync::Arc<CudaStream>,
    attention: AttentionModule,
    lm_head: LmHeadModule,
    mlp: MlpModule,
    f16: F16TcMatmulModule,
    tma: Nvfp4GemmModule,
    scale_pack: Sm120ScalePackModule,
    pad: TmaMatrixPadModule,
    rows: usize,
    k: usize,
    n: usize,
    input_bytes: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u8>,
    input_csf_scales: DeviceBuffer<u8>,
    input_globals: DeviceBuffer<f32>,
    weight_bytes: DeviceBuffer<u8>,
    weight_scales: DeviceBuffer<u8>,
    weight_csf_scales: DeviceBuffer<u8>,
    weight_primary_scales: DeviceBuffer<u8>,
    weight_global: DeviceBuffer<f32>,
    bias_bytes: DeviceBuffer<u8>,
    bias_scales: DeviceBuffer<u8>,
    bias_global: DeviceBuffer<f32>,
}

impl Fixture {
    fn new(rows: usize, k: usize, n: usize) -> Result<Self, Box<dyn Error>> {
        let (_, stream, ptx) = common::cuda_test_context()?;
        Ok(Self {
            attention: AttentionModule::from_module(ptx.clone())?,
            lm_head: LmHeadModule::from_module(ptx.clone())?,
            mlp: MlpModule::from_module(ptx.clone())?,
            f16: F16TcMatmulModule::from_module(ptx.clone())?,
            tma: Nvfp4GemmModule::from_module(ptx.clone())?,
            scale_pack: Sm120ScalePackModule::from_module(ptx.clone())?,
            pad: TmaMatrixPadModule::from_module(ptx.clone())?,
            input_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(rows, k, 13, 7, 0))?,
            input_scales: DeviceBuffer::from_host(&stream, &pattern_scales(rows * k, 1))?,
            input_csf_scales: DeviceBuffer::from_host(
                &stream,
                &common::encode_csf_scales(&pattern_scales(rows * k, 1), rows, k),
            )?,
            input_globals: DeviceBuffer::from_host(&stream, &row_globals(rows))?,
            weight_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(n, k, 11, 5, 3))?,
            weight_scales: DeviceBuffer::from_host(&stream, &pattern_scales(n * k, 2))?,
            weight_csf_scales: DeviceBuffer::from_host(
                &stream,
                &common::encode_csf_scales(&pattern_scales(n * k, 2), n, k),
            )?,
            weight_primary_scales: DeviceBuffer::from_host(
                &stream,
                &rust_kernels_cuda::nvfp4::csf::encode_host(&pattern_scales(n * k, 2), n, k),
            )?,
            weight_global: DeviceBuffer::from_host(&stream, &[0.75_f32])?,
            bias_bytes: DeviceBuffer::from_host(&stream, &sparse_bytes(1, n, 3, 1, 2))?,
            bias_scales: DeviceBuffer::from_host(
                &stream,
                &rust_kernels_cuda::nvfp4::csf::encode_host(
                    &pattern_scales(n, 3),
                    1,
                    n.max(128).next_power_of_two(),
                ),
            )?,
            bias_global: DeviceBuffer::from_host(&stream, &[0.25_f32])?,
            stream,
            rows,
            k,
            n,
        })
    }

    fn input(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.input_bytes, &self.input_scales, &self.input_globals)
    }

    fn tma_rowwise_exact(
        &self,
        out: &mut DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;
        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
                &self.stream,
                &descriptors,
                out,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
                route,
            )?;
        Ok(())
    }

    fn weight_mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(
            &self.weight_bytes,
            &self.weight_primary_scales,
            &self.weight_global,
        )
    }

    fn bias_device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bias_bytes, &self.bias_scales, &self.bias_global)
    }

    fn tma_raw(&self, raw: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed =
            DeviceBuffer::zeroed(&self.stream, sm120_scale_packed_len(padded_n, self.k))?;
        let mut weight_bytes_padded = DeviceBuffer::zeroed(&self.stream, padded_n * self.k / 2)?;
        let weight_bytes = if padded_n == self.n {
            &self.weight_bytes
        } else {
            self.pad.pad_u4_rows(U4RowPadArgs {
                stream: &self.stream,
                input: &self.weight_bytes,
                output: &mut weight_bytes_padded,
                rows: self.n as u32,
                padded_rows: padded_n as u32,
                cols_u4: self.k as u32,
            })?;
            &weight_bytes_padded
        };
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_padded_output(
            &self.stream,
            &descriptors,
            raw,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            padded_n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }

    fn tma_scalar_with_output_amax(
        &self,
        plain: &mut DeviceBuffer<f32>,
        fused: &mut DeviceBuffer<f32>,
        output_amax: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_with_output_amax(
                &self.stream,
                &descriptors,
                fused,
                output_amax,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
            )?;
        Ok(())
    }

    fn tma_relu2_backward_f16_with_output_amax(
        &self,
        pre_activation: &DeviceBuffer<u16>,
        plain: &mut DeviceBuffer<f32>,
        fused: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, Box<dyn Error>> {
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut weight_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.n), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.scale_pack.pack(
            &self.stream,
            &self.weight_scales,
            &mut weight_scale_packed,
            self.n as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.weight_bytes,
            &weight_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
                None,
            )?;
        let input_csf = DeviceBuffer::from_host(
            &self.stream,
            &common::encode_csf_scales(
                &self.input_scales.to_host_vec(&self.stream)?,
                self.rows,
                self.k,
            ),
        )?;
        let weight_csf = DeviceBuffer::from_host(
            &self.stream,
            &common::encode_csf_scales(
                &self.weight_scales.to_host_vec(&self.stream)?,
                self.n,
                self.k,
            ),
        )?;
        self.tma.prepare_tma_nvfp4_csf_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_csf,
            &self.weight_bytes,
            &weight_csf,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        Ok(self
            .tma
            .gemm_tma_nvfp4_rowwise_a_scale_relu2_backward_f16_with_output_amax(
                &self.stream,
                &descriptors,
                pre_activation,
                fused,
                chunk_amax,
                self.rows as u32,
                self.k as u32,
                self.n as u32,
                &self.input_globals,
                &self.weight_global,
                None,
            )?)
    }

    fn tma_self_symmetric_with_output_amax(
        &self,
        plain: &mut DeviceBuffer<f32>,
        symmetric: &mut DeviceBuffer<f32>,
        output_amax: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        assert_eq!(self.rows, self.n);
        let mut input_scale_packed = DeviceBuffer::zeroed(
            &self.stream,
            sm120_scale_packed_len(sm120_scale_padded_mn_extent(self.rows), self.k),
        )?;
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.scale_pack.pack(
            &self.stream,
            &self.input_scales,
            &mut input_scale_packed,
            self.rows as u32,
            self.k as u32,
        )?;
        self.tma.prepare_tma_nvfp4_device_scales_into(
            &self.stream,
            &self.input_bytes,
            &input_scale_packed,
            &self.input_bytes,
            &input_scale_packed,
            self.rows as u32,
            self.k as u32,
            self.rows as u32,
            &mut descriptors,
        )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers(
                &self.stream,
                &descriptors,
                plain,
                self.rows as u32,
                self.k as u32,
                self.rows as u32,
                &self.input_globals,
                &self.input_globals,
            )?;
        self.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_symmetric_with_precleared_output_amax(
                &self.stream,
                &descriptors,
                symmetric,
                output_amax,
                self.rows as u32,
                self.k as u32,
                &self.input_globals,
            )?;
        Ok(())
    }

    fn tma_affine(&self, out: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        self.tma_affine_columns(out, None)
    }

    fn tma_affine_columns(
        &self,
        out: &mut DeviceBuffer<f32>,
        columns: Option<std::ops::Range<u32>>,
    ) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);

        let mut weight_bytes_padded = DeviceBuffer::zeroed(&self.stream, padded_n * self.k / 2)?;
        let weight_bytes = if padded_n == self.n {
            &self.weight_bytes
        } else {
            self.pad.pad_u4_rows(U4RowPadArgs {
                stream: &self.stream,
                input: &self.weight_bytes,
                output: &mut weight_bytes_padded,
                rows: self.n as u32,
                padded_rows: padded_n as u32,
                cols_u4: self.k as u32,
            })?;
            &weight_bytes_padded
        };
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.tma.prepare_tma_nvfp4_csf_scales_into(
            &self.stream,
            &self.input_bytes,
            &self.input_csf_scales,
            weight_bytes,
            &self.weight_csf_scales,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_affine_columns(
            &self.stream,
            &descriptors,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            padded_n as u32,
            &self.input_globals,
            &self.weight_global,
            columns,
        )?;
        Ok(())
    }

    fn old_raw(&self, out: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        self.lm_head.logits(LmHeadArgs {
            stream: &self.stream,
            input: self.input(),
            weight: self.weight_mma(),
            logits: out,
            token_count: self.rows as u32,
            input_dim: self.k as u32,
            vocab_size: self.n as u32,
        })?;
        Ok(())
    }

    fn tma_residual(&self, residual: &mut DeviceBuffer<f32>) -> Result<(), Box<dyn Error>> {
        self.tma_projection(residual, None, true)
    }

    fn tma_projection(
        &self,
        out: &mut DeviceBuffer<f32>,
        route: Option<Nvfp4GemmRoute<'_>>,
        add_residual: bool,
    ) -> Result<(), Box<dyn Error>> {
        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.tma.prepare_tma_nvfp4_csf_scales_into(
            &self.stream,
            &self.input_bytes,
            &self.input_csf_scales,
            &self.weight_bytes,
            &self.weight_csf_scales,
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_projection(
            &self.stream,
            &descriptors,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
            route,
            add_residual,
        )?;
        Ok(())
    }

    fn tma_relu2(
        &self,
        pre_activation: &mut DeviceBuffer<f32>,
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        assert_eq!(padded_n, self.n, "fused ReLU2 requires exact output tiles");

        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.tma.prepare_tma_nvfp4_csf_scales_into(
            &self.stream,
            &self.input_bytes,
            &self.input_csf_scales,
            &self.weight_bytes,
            &self.weight_csf_scales,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_relu2(
            &self.stream,
            &descriptors,
            pre_activation,
            pre_activation_f16,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }

    fn tma_relu2_compact(
        &self,
        pre_activation_f16: Option<&mut DeviceBuffer<u16>>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>> {
        let padded_n = sm120_scale_padded_mn_extent(self.n);
        assert_eq!(padded_n, self.n, "fused ReLU2 requires exact output tiles");

        let mut descriptors = TmaNvfp4DeviceScaleDescriptors::new(&self.stream)?;

        self.tma.prepare_tma_nvfp4_csf_scales_into(
            &self.stream,
            &self.input_bytes,
            &self.input_csf_scales,
            &self.weight_bytes,
            &self.weight_csf_scales,
            self.rows as u32,
            self.k as u32,
            padded_n as u32,
            &mut descriptors,
        )?;
        self.tma.gemm_tma_nvfp4_rowwise_a_scale_relu2_compact(
            &self.stream,
            &descriptors,
            pre_activation_f16,
            out,
            self.bias_device(),
            self.rows as u32,
            self.k as u32,
            self.n as u32,
            &self.input_globals,
            &self.weight_global,
        )?;
        Ok(())
    }
}

fn sparse_bytes(rows: usize, cols: usize, row_mul: usize, col_mul: usize, add: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows * cols / 2];
    for row in 0..rows {
        for col in 0..cols {
            if (row * row_mul + col * col_mul + add) % 17 == 0 {
                set_e2m1_one(&mut bytes, row * cols + col);
            }
        }
    }
    bytes
}

fn pattern_scales(elements: usize, offset: usize) -> Vec<u8> {
    let mut scales = one_scales(elements);
    for (index, scale) in scales.iter_mut().enumerate() {
        *scale = [0x30, 0x34, 0x38, 0x3c, 0x40][(index + offset) % 5];
    }
    scales
}

fn row_globals(rows: usize) -> Vec<f32> {
    (0..rows)
        .map(|row| 0.5 + (row % 7) as f32 * 0.125)
        .collect()
}

fn residual_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 23) as f32 * 0.03125 - 0.25)
        .collect()
}

fn action_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 29) as f32 * -0.0234375 + 0.375)
        .collect()
}

fn pre_activation_values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % 31) as f32 * 0.03125 - 0.4375)
        .collect()
}

// Decode the fixture payload and group scales independently of the CUDA packer.
fn fixture_dot(row: usize, col: usize, k: usize, self_product: bool) -> f32 {
    let scales = [0.5_f32, 0.75, 1.0, 1.5, 2.0];
    let mut sum = 0.0;
    for i in 0..k {
        let a = if (row * 13 + i * 7) % 17 == 0 {
            scales[(row * (k / 16) + i / 16 + 1) % 5]
        } else {
            0.0
        };
        let b = if self_product {
            if (col * 13 + i * 7) % 17 == 0 {
                scales[(col * (k / 16) + i / 16 + 1) % 5]
            } else {
                0.0
            }
        } else if (col * 11 + i * 5 + 3) % 17 == 0 {
            scales[(col * (k / 16) + i / 16 + 2) % 5]
        } else {
            0.0
        };
        sum += a * b;
    }
    sum * if self_product { 0.25 } else { 0.375 }
}

#[test]
#[ignore = "requires generated sm_120a PTX"]
fn tma_polar_epilogues_match_decoded_fp4_and_guard_compact_output() -> Result<(), Box<dyn Error>> {
    let f = Fixture::new(256, 256, 256)?;
    let mut a_scales = DeviceBuffer::zeroed(&f.stream, sm120_scale_packed_len(256, 256))?;
    let mut b_scales = DeviceBuffer::zeroed(&f.stream, sm120_scale_packed_len(256, 256))?;
    f.scale_pack
        .pack(&f.stream, &f.input_scales, &mut a_scales, 256, 256)?;
    f.scale_pack
        .pack(&f.stream, &f.weight_scales, &mut b_scales, 256, 256)?;
    let mut tma = TmaNvfp4DeviceScaleDescriptors::new(&f.stream)?;
    // Reuse the same operand addresses and shape across descriptor geometries.
    // A stale projection-sized box would overrun the polar shared-memory tile.
    f.tma.prepare_tma_nvfp4_device_scales_into(
        &f.stream,
        &f.input_bytes,
        &a_scales,
        &f.weight_bytes,
        &b_scales,
        256,
        256,
        256,
        &mut tma,
    )?;
    for &(rows, cols) in &[(128_usize, 160_usize), (256, 256)] {
        for &(bound, apply) in &[(0.25_f32, true), (1.5625, true), (1.5625, false)] {
            f.tma.prepare_tma_nvfp4_polar_scales_into(
                &f.stream,
                &f.input_bytes,
                &a_scales,
                &f.weight_bytes,
                &b_scales,
                256,
                256,
                256,
                &mut tma,
            )?;
            let values = residual_values(rows * cols);
            let source = DeviceBuffer::from_host(&f.stream, &values)?;
            let bound_gpu = DeviceBuffer::from_host(&f.stream, &[bound])?;
            let mut amax = DeviceBuffer::zeroed(&f.stream, 1)?;
            let mut out = DeviceBuffer::from_host(&f.stream, &vec![12345.0_f32; rows * cols + 64])?;
            let mut output_bound = DeviceBuffer::<f32>::zeroed(&f.stream, 1)?;
            let mut partials =
                DeviceBuffer::zeroed(&f.stream, rows * (256 / POLAR_TILE_N as usize))?;
            let inv = if apply && bound > 1.0 {
                1.0 / bound.sqrt()
            } else {
                1.0
            };
            let expected: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    3.4445_f32.mul_add(values[i] * inv, fixture_dot(i / cols, i % cols, 256, false))
                })
                .collect();
            f.tma
                .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_polar_step_with_precleared_output_amax(
                    &f.stream,
                    &tma,
                    &source,
                    &mut out,
                    &bound_gpu,
                    &mut amax,
                    256,
                    256,
                    256,
                    &f.input_globals,
                    &f.weight_global,
                    3.4445,
                    rows as u32,
                    cols as u32,
                    apply,
                )?;
            let got = out.to_host_vec(&f.stream)?;
            assert_eq!(&got[..rows * cols], &expected);
            assert!(got[rows * cols..].iter().all(|x| *x == 12345.0));
            assert_eq!(
                amax.to_host_vec(&f.stream)?[0],
                expected
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0_f32, f32::max)
            );
            f.tma
                .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_polar_step_with_precleared_row_sumsq(
                    &f.stream,
                    &tma,
                    &source,
                    &mut out,
                    &bound_gpu,
                    &mut output_bound,
                    &mut partials,
                    256,
                    256,
                    256,
                    &f.input_globals,
                    &f.weight_global,
                    3.4445,
                    rows as u32,
                    cols as u32,
                    apply,
                )?;
            let got = out.to_host_vec(&f.stream)?;
            assert_eq!(&got[..rows * cols], &expected);
            assert!(got[rows * cols..].iter().all(|x| *x == 12345.0));
            let norm = expected
                .chunks_exact(cols)
                .map(|row| row.iter().fold(0.0_f64, |s, v| s + (*v as f64).powi(2)))
                .fold(0.0_f64, f64::max);
            let actual = output_bound.to_host_vec(&f.stream)?[0] as f64;
            assert_eq!(bound_gpu.to_host_vec(&f.stream)?, vec![bound]);
            assert!(
                (actual - norm).abs() <= norm.max(1.0) * 2.0e-6,
                "{actual} vs {norm}"
            );
        }
    }
    // Three triangular CTAs include the mirrored off-diagonal tile. The input
    // G is symmetric; its packed group scales intentionally vary by row.
    f.tma.prepare_tma_nvfp4_device_scales_into(
        &f.stream,
        &f.input_bytes,
        &a_scales,
        &f.input_bytes,
        &a_scales,
        256,
        256,
        256,
        &mut tma,
    )?;
    let gram: Vec<f32> = (0..256 * 256)
        .map(|i| ((i / 256 + i % 256) % 23) as f32 / 32.0)
        .collect();
    let gram_gpu = DeviceBuffer::from_host(&f.stream, &gram)?;
    for &(dim, bound, apply) in &[
        (256_usize, 1.5625_f32, true),
        (160, 1.5625, false),
        (160, 0.25, true),
    ] {
        // The compact source is intentionally rebuilt for the partial tile.
        let compact: Vec<f32> = (0..dim * dim)
            .map(|i| ((i / dim + i % dim) % 23) as f32 / 32.0)
            .collect();
        let compact_gpu = DeviceBuffer::from_host(&f.stream, &compact)?;
        let mut out = DeviceBuffer::from_host(&f.stream, &vec![12345.0_f32; dim * dim + 64])?;
        let mut amax = DeviceBuffer::zeroed(&f.stream, 1)?;
        let bound_gpu = DeviceBuffer::from_host(&f.stream, &[bound])?;
        f.tma
            .gemm_tma_nvfp4_device_scales_and_global_scale_buffers_gram_factor_with_precleared_output_amax(
                &f.stream,
                &tma,
                if dim == 256 { &gram_gpu } else { &compact_gpu },
                &bound_gpu,
                dim as u32,
                -4.775,
                2.0315,
                apply,
                &mut out,
                &mut amax,
                256,
                256,
                &f.input_globals,
            )?;
        let got = out.to_host_vec(&f.stream)?;
        let inv = if apply && bound > 1.0 {
            1.0 / bound
        } else {
            1.0
        };
        let expected: Vec<f32> = (0..dim * dim)
            .map(|i| {
                (-4.775_f32).mul_add(
                    compact[i] * inv,
                    2.0315 * fixture_dot(i / dim, i % dim, 256, true),
                )
            })
            .collect();
        assert_eq!(&got[..dim * dim], &expected);
        assert!(got[dim * dim..].iter().all(|x| *x == 12345.0));
        assert_eq!(
            amax.to_host_vec(&f.stream)?[0],
            expected
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max)
        );
    }
    Ok(())
}
