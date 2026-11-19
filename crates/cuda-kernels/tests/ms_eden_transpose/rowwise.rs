use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4RowwiseDecodeTransposeArgs};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

use super::TestResult;
use super::common;
use super::support::*;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn rowwise_nvfp4_transpose_ms_eden_matches_materialized_decode() -> TestResult {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = input_matrix();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = RowwiseSourceScratch::new(&stream)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;
    let mut materialized = QuantScratch::new(&stream)?;
    let mut direct = QuantScratch::new(&stream)?;

    source.quantize(&stream, &quant, &x_dev, ROWS, COLS)?;

    let source_tensor = source.tensor();
    decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: ROWS as u32,
        cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale(
        materialized.quartet_args(&stream, &x_t_dev, COLS, ROWS, padded_rows()),
    )?;
    quant.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        direct.rowwise_transpose_args(&stream, source_tensor, ROWS, COLS, padded_rows()),
    )?;

    direct.assert_quartet_eq(&stream, &materialized)?;
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn rowwise_nvfp4_transpose_no_chunk_no_pad_matches_materialized_decode() -> TestResult {
    const LOCAL_ROWS: usize = 128;
    const LOCAL_COLS: usize = 128;

    let (_, stream, ptx) = common::cuda_test_context()?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let x = (0..LOCAL_ROWS * LOCAL_COLS)
        .map(|index| {
            let row = index / LOCAL_COLS;
            let col = index % LOCAL_COLS;
            (row as f32 - 12.0) * 0.015625 + (col as f32 - 5.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut source = RowwiseSourceScratch::new_for_shape(&stream, LOCAL_ROWS, LOCAL_COLS)?;
    let mut x_t_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;
    let mut materialized = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;
    let mut direct = QuantScratch::new_exact(&stream, LOCAL_COLS, LOCAL_ROWS)?;

    source.quantize(&stream, &quant, &x_dev, LOCAL_ROWS, LOCAL_COLS)?;

    let source_tensor = source.tensor();
    decode.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
        stream: &stream,
        input: source_tensor,
        output: &mut x_t_dev,
        rows: LOCAL_ROWS as u32,
        cols: LOCAL_COLS as u32,
    })?;
    quant.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        materialized.quartet_args(&stream, &x_t_dev, LOCAL_COLS, LOCAL_ROWS, LOCAL_ROWS),
    )?;
    quant.rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        direct.rowwise_transpose_args(&stream, source_tensor, LOCAL_ROWS, LOCAL_COLS, LOCAL_ROWS),
    )?;

    direct.assert_no_chunk_quartet_eq(&stream, &materialized)?;
    Ok(())
}
