use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenQuantArgs, Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4QuantPaddedArgs,
    Nvfp4QuantTransposePaddedArgs,
};
use rust_kernels_cuda::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

mod common;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_to_nvfp4_four_six_writes_quantized_outputs() -> Result<(), Box<dyn Error>> {
    let x = [
        -3.25f32, -2.0, -1.25, -0.5, -0.125, 0.0, 0.25, 0.75, 1.0, 1.5, 2.25, 3.0, 4.0, 5.0, 6.5,
        8.0,
    ];
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut fp4_dev,
        out_scales: &mut scales_dev,
        out_global_scale: &mut global_scale_dev,
        group_count: 1,
    })?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scale = global_scale_dev.to_host_vec(&stream)?;

    common::assert_nvfp4_buffers_nonzero(&fp4, &scales);
    assert!((global_scale[0] - 8.0 / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn tiled_four_six_transpose_matches_explicit_transpose() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 16;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| {
            let row = index / COLS;
            let col = index % COLS;
            ((row * 17 + col * 13) as f32 - 512.0) * 0.03125
        })
        .collect::<Vec<_>>();
    let mut x_t = vec![0.0f32; x.len()];
    for row in 0..ROWS {
        for col in 0..COLS {
            x_t[col * ROWS + row] = x[row * COLS + col];
        }
    }
    let amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let x_t_dev = DeviceBuffer::from_host(&stream, &x_t)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax)?;
    let mut tiled_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut tiled_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut tiled_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    module.fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &x_dev,
        amax: &amax_dev,
        out_fp4: &mut tiled_fp4,
        out_scales: &mut tiled_scales,
        out_global_scale: &mut tiled_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;
    module.fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &x_t_dev,
        amax: &amax_dev,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        rows: COLS as u32,
        cols: ROWS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;

    assert_eq!(
        tiled_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        tiled_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        tiled_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn fp32_to_nvfp4_ms_eden_writes_rotated_quantized_outputs() -> Result<(), Box<dyn Error>> {
    let x = (0..64)
        .map(|index| (index as f32 - 31.5) * 0.03125)
        .collect::<Vec<_>>();

    let (_, stream, module) = common::cuda_test_module(Nvfp4QuantModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut scales_dev = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut global_scales_dev = DeviceBuffer::<f32>::zeroed(&stream, 2)?;
    let mut chunk_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len() / 32)?;

    module.fp32_to_nvfp4_ms_eden(MsEdenQuantArgs {
        stream: &stream,
        x: &x_dev,
        out_fp4: &mut fp4_dev,
        out_scales: &mut scales_dev,
        out_global_scales: &mut global_scales_dev,
        out_chunk_amax: &mut chunk_amax_dev,
        row_count: 2,
        src_row_len: 32,
        dst_row_len: 32,
        global_scale: 1.0,
        scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
        sign_seed: 0x1234_5678,
        scale_seed: 0x9abc_def0,
    })?;

    let fp4 = fp4_dev.to_host_vec(&stream)?;
    let scales = scales_dev.to_host_vec(&stream)?;
    let global_scales = global_scales_dev.to_host_vec(&stream)?;
    let chunk_amax = chunk_amax_dev.to_host_vec(&stream)?;

    common::assert_nvfp4_buffers_nonzero(&fp4, &scales);
    assert!(
        global_scales
            .iter()
            .all(|scale| (*scale - 1.0).abs() <= 1.0e-8)
    );
    assert!(
        chunk_amax
            .iter()
            .all(|amax| *amax > 0.0 && amax.is_finite())
    );
    Ok(())
}
