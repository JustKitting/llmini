use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, MsEdenPairDeviceScaleQuantArgs,
    MsEdenTransposeDeviceScaleQuantArgs, Nvfp4QuantModule,
};
use rust_kernels_cuda::transpose::{TransposeF32Args, TransposeModule};

use super::TestResult;
use super::common;
use super::support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_transpose_ms_eden_matches_materialized_transpose() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let transpose = TransposeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let global_scale = DeviceBuffer::from_host(&stream, &[0.75_f32])?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    transpose.transpose_f32(TransposeF32Args {
        stream: &stream,
        input: &x_dev,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;

    quant.fp32_to_nvfp4_ms_eden_device_scale(MsEdenDeviceScaleQuantArgs {
        stream: &stream,
        x: &x_t_dev,
        out_fp4: &mut materialized.bytes,
        out_scales: &mut materialized.scales,
        out_global_scales: &mut materialized.global_scales,
        out_chunk_amax: &mut materialized.chunk_amax,
        global_scale: &global_scale,
        row_count: COLS as u32,
        src_row_len: ROWS as u32,
        dst_row_len: padded_rows() as u32,
        scale_override: SCALE_OVERRIDE,
        sign_seed: SIGN_SEED,
        scale_seed: SCALE_SEED,
    })?;

    quant.fp32_transpose_to_nvfp4_ms_eden_device_scale(MsEdenTransposeDeviceScaleQuantArgs {
        stream: &stream,
        x: &x_dev,
        out_fp4: &mut direct.bytes,
        out_scales: &mut direct.scales,
        out_global_scales: &mut direct.global_scales,
        out_chunk_amax: &mut direct.chunk_amax,
        global_scale: &global_scale,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        dst_row_len: padded_rows() as u32,
        scale_override: SCALE_OVERRIDE,
        sign_seed: SIGN_SEED,
        scale_seed: SCALE_SEED,
    })?;

    direct.assert_ms_eden_eq(&stream, &materialized)?;
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_pair_tiled_matches_independent_row_and_transpose_quantization() -> TestResult {
    const LOCAL_ROWS: usize = 64;
    const LOCAL_COLS: usize = 32;
    const TRANSPOSE_SCALE_SEED: u32 = SCALE_SEED ^ 0x85eb_ca6b;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let x = (0..LOCAL_ROWS * LOCAL_COLS)
        .map(|index| {
            let row = index / LOCAL_COLS;
            let col = index % LOCAL_COLS;
            (row as f32 - 21.0) * 0.0078125 + (col as f32 - 9.0) * 0.015625
        })
        .collect::<Vec<_>>();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut pair_row = QuantScratch::new_exact(&stream, LOCAL_ROWS, LOCAL_COLS)?;
    let mut pair_transpose = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;
    let mut row_reference = QuantScratch::new_exact(&stream, LOCAL_ROWS, LOCAL_COLS)?;
    let mut transpose_reference = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;
    let mut bias = DeviceBuffer::<f32>::zeroed(&stream, LOCAL_COLS)?;

    let bias_fused = quant
        .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_with_bias(
            MsEdenPairDeviceScaleQuantArgs {
                stream: &stream,
                x: &x_dev,
                out_fp4: &mut pair_row.bytes,
                out_scales: &mut pair_row.scales,
                out_global_scales: &mut pair_row.global_scales,
                transpose_out_fp4: &mut pair_transpose.bytes,
                transpose_out_scales: &mut pair_transpose.scales,
                transpose_out_global_scales: &mut pair_transpose.global_scales,
                out_chunk_amax: &mut pair_row.chunk_amax,
                out_global_scale: &mut pair_row.global_scale,
                row_count: LOCAL_ROWS as u32,
                src_row_len: LOCAL_COLS as u32,
                dst_row_len: LOCAL_COLS as u32,
                transpose_dst_row_len: LOCAL_ROWS as u32,
                scale_override: SCALE_OVERRIDE,
                sign_seed: SIGN_SEED,
                scale_seed: SCALE_SEED,
                transpose_scale_seed: TRANSPOSE_SCALE_SEED,
                precomputed_chunk_count: None,
            },
            &mut bias,
        )?;

    quant.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(MsEdenDeviceScaleQuantArgs {
        stream: &stream,
        x: &x_dev,
        out_fp4: &mut row_reference.bytes,
        out_scales: &mut row_reference.scales,
        out_global_scales: &mut row_reference.global_scales,
        out_chunk_amax: &mut row_reference.chunk_amax,
        global_scale: &pair_row.global_scale,
        row_count: LOCAL_ROWS as u32,
        src_row_len: LOCAL_COLS as u32,
        dst_row_len: LOCAL_COLS as u32,
        scale_override: SCALE_OVERRIDE,
        sign_seed: SIGN_SEED,
        scale_seed: SCALE_SEED,
    })?;
    quant.fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
        MsEdenTransposeDeviceScaleQuantArgs {
            stream: &stream,
            x: &x_dev,
            out_fp4: &mut transpose_reference.bytes,
            out_scales: &mut transpose_reference.scales,
            out_global_scales: &mut transpose_reference.global_scales,
            out_chunk_amax: &mut transpose_reference.chunk_amax,
            global_scale: &pair_row.global_scale,
            source_rows: LOCAL_ROWS as u32,
            source_cols: LOCAL_COLS as u32,
            dst_row_len: LOCAL_ROWS as u32,
            scale_override: SCALE_OVERRIDE,
            sign_seed: SIGN_SEED,
            scale_seed: TRANSPOSE_SCALE_SEED,
        },
    )?;

    pair_row.assert_payload_eq(&stream, &row_reference)?;
    pair_transpose.assert_payload_eq(&stream, &transpose_reference)?;
    assert!(bias_fused);
    let expected_bias = (0..LOCAL_COLS)
        .map(|col| (0..LOCAL_ROWS).map(|row| x[row * LOCAL_COLS + col]).sum())
        .collect::<Vec<f32>>();
    common::assert_slice_close(&bias.to_host_vec(&stream)?, &expected_bias, 1.0e-5);
    Ok(())
}
