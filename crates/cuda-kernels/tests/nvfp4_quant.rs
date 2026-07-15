use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f32_matrix_ops::{F32MatrixOpsModule, F32ScaleInPlaceByAmaxArgs};
use rust_kernels_cuda::nvfp4_quant::{
    MsEdenQuantArgs, Nvfp4QuantArgs, Nvfp4QuantModule, Nvfp4QuantPaddedArgs,
    Nvfp4QuantTransposePaddedArgs, TensorAmaxArgs, nvfp4_tensor_amax_chunks,
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
fn bounded_amax_four_six_matches_rescanned_input() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 64;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| ((index % 257) as f32 - 128.0) * 0.137)
        .collect::<Vec<_>>();
    let original_amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let mut bounded = DeviceBuffer::from_host(&stream, &x)?;
    let original_amax_dev = DeviceBuffer::from_host(&stream, &original_amax)?;
    f32_ops.scale_in_place_by_amax_bound(F32ScaleInPlaceByAmaxArgs {
        stream: &stream,
        x: &mut bounded,
        amax: &original_amax_dev,
        len: x.len() as u32,
    })?;

    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, nvfp4_tensor_amax_chunks(x.len()))?;
    let mut rescanned_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    quant.tensor_amax_f32(TensorAmaxArgs {
        stream: &stream,
        x: &bounded,
        chunk_amax: &mut chunk_amax,
        out: &mut rescanned_amax,
        element_count: x.len() as u32,
    })?;

    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let unbounded = DeviceBuffer::from_host(&stream, &x)?;
    let mut lazy_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut lazy_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut lazy_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    quant.fp32_to_nvfp4_four_six_padded(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &rescanned_amax,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_four_six_exact_bounded_amax(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &original_amax_dev,
        out_fp4: &mut bounded_fp4,
        out_scales: &mut bounded_scales,
        out_global_scale: &mut bounded_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;
    quant.fp32_to_nvfp4_four_six_exact_lazy_bounded_amax(Nvfp4QuantPaddedArgs {
        stream: &stream,
        x: &unbounded,
        amax: &original_amax_dev,
        out_fp4: &mut lazy_fp4,
        out_scales: &mut lazy_scales,
        out_global_scale: &mut lazy_global,
        rows: ROWS as u32,
        cols: COLS as u32,
        padded_rows: ROWS as u32,
        padded_cols: COLS as u32,
    })?;

    assert_eq!(
        bounded_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
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
fn sqrt_bounded_amax_transpose_matches_rescanned_input() -> Result<(), Box<dyn Error>> {
    const ROWS: usize = 16;
    const COLS: usize = 64;
    let x = (0..ROWS * COLS)
        .map(|index| ((index % 173) as f32 - 86.0) * 0.119)
        .collect::<Vec<_>>();
    let original_amax = [x.iter().fold(0.0f32, |max, value| max.max(value.abs()))];
    let sqrt_bound_amax = [3.7_f32];

    let (_, stream, ptx) = common::cuda_test_context()?;
    let quant = Nvfp4QuantModule::from_module(ptx.clone())?;
    let f32_ops = F32MatrixOpsModule::from_module(ptx)?;
    let mut bounded = DeviceBuffer::from_host(&stream, &x)?;
    let original_amax_dev = DeviceBuffer::from_host(&stream, &original_amax)?;
    let sqrt_bound_amax_dev = DeviceBuffer::from_host(&stream, &sqrt_bound_amax)?;
    f32_ops.scale_in_place_by_sqrt_amax_bound(F32ScaleInPlaceByAmaxArgs {
        stream: &stream,
        x: &mut bounded,
        amax: &sqrt_bound_amax_dev,
        len: x.len() as u32,
    })?;

    let mut chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, nvfp4_tensor_amax_chunks(x.len()))?;
    let mut rescanned_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    quant.tensor_amax_f32(TensorAmaxArgs {
        stream: &stream,
        x: &bounded,
        chunk_amax: &mut chunk_amax,
        out: &mut rescanned_amax,
        element_count: x.len() as u32,
    })?;

    let mut reference_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut reference_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut reference_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut bounded_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut bounded_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut bounded_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let unbounded = DeviceBuffer::from_host(&stream, &x)?;
    let mut lazy_fp4 = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 2)?;
    let mut lazy_scales = DeviceBuffer::<u8>::zeroed(&stream, x.len() / 16)?;
    let mut lazy_global = DeviceBuffer::<f32>::zeroed(&stream, 1)?;

    quant.fp32_transpose_to_nvfp4_four_six_padded(Nvfp4QuantTransposePaddedArgs {
        stream: &stream,
        x: &bounded,
        amax: &rescanned_amax,
        out_fp4: &mut reference_fp4,
        out_scales: &mut reference_scales,
        out_global_scale: &mut reference_global,
        source_rows: ROWS as u32,
        source_cols: COLS as u32,
        padded_rows: COLS as u32,
        padded_cols: ROWS as u32,
    })?;
    quant.fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
        Nvfp4QuantTransposePaddedArgs {
            stream: &stream,
            x: &bounded,
            amax: &original_amax_dev,
            out_fp4: &mut bounded_fp4,
            out_scales: &mut bounded_scales,
            out_global_scale: &mut bounded_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            padded_rows: COLS as u32,
            padded_cols: ROWS as u32,
        },
        &sqrt_bound_amax_dev,
    )?;
    quant.fp32_transpose_to_nvfp4_four_six_exact_lazy_sqrt_bounded_amax(
        Nvfp4QuantTransposePaddedArgs {
            stream: &stream,
            x: &unbounded,
            amax: &original_amax_dev,
            out_fp4: &mut lazy_fp4,
            out_scales: &mut lazy_scales,
            out_global_scale: &mut lazy_global,
            source_rows: ROWS as u32,
            source_cols: COLS as u32,
            padded_rows: COLS as u32,
            padded_cols: ROWS as u32,
        },
        &sqrt_bound_amax_dev,
    )?;

    assert_eq!(
        bounded_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        bounded_global.to_host_vec(&stream)?,
        reference_global.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_fp4.to_host_vec(&stream)?,
        reference_fp4.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_scales.to_host_vec(&stream)?,
        reference_scales.to_host_vec(&stream)?
    );
    assert_eq!(
        lazy_global.to_host_vec(&stream)?,
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
