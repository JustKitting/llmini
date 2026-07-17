use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4DecodeTransposeArgs};
use rust_kernels_cuda::nvfp4_quant::{
    Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4TransposeMsEdenDeviceScaleQuantArgs,
};

use super::TestResult;
use super::common;
use super::support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_ms_eden_matches_materialized_decode() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = SourceScratch::new(&stream)?;
    let amax = DeviceBuffer::from_host(&stream, &[cpu_amax(&x)])?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax,
        out_fp4: &mut source.bytes,
        out_scales: &mut source.scales,
        out_global_scale: &mut source.global_scale,
        group_count: (ROWS * COLS / 16) as u32,
    })?;

    let source_tensor = source.tensor();
    decode.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        materialized.quartet_args(&stream, &x_t_dev, COLS, ROWS, padded_rows()),
    )?;
    quant.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
            stream: &stream,
            input: source_tensor,
            out_fp4: &mut direct.bytes,
            out_scales: &mut direct.scales,
            out_global_scales: &mut direct.global_scales,
            out_chunk_amax: &mut direct.chunk_amax,
            out_global_scale: &mut direct.global_scale,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            dst_row_len: padded_rows() as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        },
    )?;

    direct.assert_quartet_eq(&stream, &materialized)?;
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_no_pad_pow2_tiled_matches_materialized_decode() -> TestResult {
    assert_nvfp4_transpose_no_pad_tiled_matches_materialized_decode(64, 32)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn nvfp4_transpose_no_pad_row_mul_tiled_matches_materialized_decode() -> TestResult {
    assert_nvfp4_transpose_no_pad_tiled_matches_materialized_decode(96, 32)
}

fn assert_nvfp4_transpose_no_pad_tiled_matches_materialized_decode(
    local_rows: usize,
    local_cols: usize,
) -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;
    let x = (0..local_rows * local_cols)
        .map(|index| {
            let row = index / local_cols;
            let col = index % local_cols;
            (row as f32 - 21.0) * 0.0078125 + (col as f32 - 9.0) * 0.015625
        })
        .collect::<Vec<_>>();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = SourceScratch::new_for_shape(&stream, local_rows, local_cols)?;
    let amax = DeviceBuffer::from_host(&stream, &[cpu_amax(&x)])?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, local_rows * local_cols)?;
    let mut materialized = QuantScratch::new_exact(&stream, local_cols, local_rows)?;
    let mut direct = QuantScratch::new_exact(&stream, local_cols, local_rows)?;

    quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax,
        out_fp4: &mut source.bytes,
        out_scales: &mut source.scales,
        out_global_scale: &mut source.global_scale,
        group_count: (local_rows * local_cols / 16) as u32,
    })?;

    let source_tensor = source.tensor();
    decode.decode_transpose_f32(Nvfp4DecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: local_rows as u32,
        cols: local_cols as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        materialized.quartet_args(&stream, &x_t_dev, local_cols, local_rows, local_rows),
    )?;
    quant.nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
            stream: &stream,
            input: source_tensor,
            out_fp4: &mut direct.bytes,
            out_scales: &mut direct.scales,
            out_global_scales: &mut direct.global_scales,
            out_chunk_amax: &mut direct.chunk_amax,
            out_global_scale: &mut direct.global_scale,
            source_rows: local_rows as u32,
            source_cols: local_cols as u32,
            dst_row_len: local_rows as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        },
    )?;

    direct.assert_no_chunk_quartet_eq(&stream, &materialized)?;
    Ok(())
}
