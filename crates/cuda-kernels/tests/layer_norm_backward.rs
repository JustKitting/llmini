use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardInputAddAmaxArgs, LayerNormBackwardInputAddArgs,
    LayerNormBackwardInputF32Args, LayerNormBackwardModule,
};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;
#[path = "layer_norm/stats.rs"]
mod stats;

use common::nvfp4::{one_pair_bytes, one_scales};
use stats::{reference_row_stats, sample_rows};

const ROWS: usize = 2;
const COLS: usize = 32;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_input_matches_reference() -> Result<(), Box<dyn Error>> {
    let epsilon = 1.0e-5f32;
    let output_scale = 0.25f32;
    let x = sample_rows(ROWS, COLS, 17, 8.0, 0.125, 0.25);
    let d_normalized = sample_rows(ROWS, COLS, 11, 5.0, 0.03125, 0.0);
    let (mean, inv_std) = reference_row_stats(&x, ROWS, COLS, epsilon);

    let (_, stream, module) = common::cuda_test_module(LayerNormBackwardModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let grad_dev = DeviceBuffer::from_host(&stream, &d_normalized)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &one_pair_bytes(COLS))?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(COLS))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut dx_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS * COLS)?;

    module.backward_input_f32(LayerNormBackwardInputF32Args {
        stream: &stream,
        residual: &x_dev,
        d_normalized: &grad_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        d_residual: &mut dx_dev,
        output_scale,
        row_count: ROWS as u32,
        embedding_dim: COLS as u32,
    })?;

    let dx = dx_dev.to_host_vec(&stream)?;
    let expected = reference_backward_input(&x, &d_normalized, &mean, &inv_std, output_scale);
    common::assert_slice_close(&dx, &expected, 1.0e-8);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_input_add_amax_matches_output_tail() -> Result<(), Box<dyn Error>> {
    const AMAX_ROWS: usize = 2;
    const AMAX_COLS: usize = 2048;

    let residual = vec![0_u16; AMAX_ROWS * AMAX_COLS];
    let d_normalized = vec![0.0_f32; AMAX_ROWS * AMAX_COLS];
    let direct = (0..AMAX_ROWS * AMAX_COLS)
        .map(|index| ((index % AMAX_COLS) as f32 - 383.5) * 0.001)
        .collect::<Vec<_>>();
    let mean = vec![0.0_f32; AMAX_ROWS];
    let inv_std = vec![1.0_f32; AMAX_ROWS];
    let initial = (0..AMAX_ROWS * AMAX_COLS)
        .map(|index| {
            if index % AMAX_COLS >= 768 {
                5.0 + (index % 17) as f32 * 0.125
            } else {
                -9.0
            }
        })
        .collect::<Vec<_>>();

    let (_, stream, module) = common::cuda_test_module(LayerNormBackwardModule::from_module)?;
    let residual_dev = DeviceBuffer::from_host(&stream, &residual)?;
    let grad_dev = DeviceBuffer::from_host(&stream, &d_normalized)?;
    let direct_dev = DeviceBuffer::from_host(&stream, &direct)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &one_pair_bytes(AMAX_COLS))?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(AMAX_COLS))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut reference_dev = DeviceBuffer::from_host(&stream, &initial)?;
    let mut candidate_dev = DeviceBuffer::from_host(&stream, &initial)?;
    let mut chunk_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, AMAX_ROWS)?;

    module.backward_input_add(LayerNormBackwardInputAddArgs {
        stream: &stream,
        residual: &residual_dev,
        d_normalized: &grad_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        direct: &direct_dev,
        d_residual: &mut reference_dev,
        output_scale: 0.25,
        row_count: AMAX_ROWS as u32,
        embedding_dim: AMAX_COLS as u32,
    })?;
    let chunk_count = module.backward_input_add_amax(LayerNormBackwardInputAddAmaxArgs {
        stream: &stream,
        residual: &residual_dev,
        d_normalized: &grad_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        direct: &direct_dev,
        d_residual: &mut candidate_dev,
        chunk_amax: &mut chunk_amax_dev,
        output_scale: 0.25,
        row_count: AMAX_ROWS as u32,
        embedding_dim: AMAX_COLS as u32,
    })?;

    let reference = reference_dev.to_host_vec(&stream)?;
    let candidate = candidate_dev.to_host_vec(&stream)?;
    assert_eq!(candidate, reference);
    assert_eq!(chunk_count, AMAX_ROWS as u32);
    for (row, got) in chunk_amax_dev.to_host_vec(&stream)?.into_iter().enumerate() {
        let expected = reference[row * AMAX_COLS..(row + 1) * AMAX_COLS]
            .iter()
            .fold(0.0_f32, |amax, value| amax.max(value.abs()));
        assert_eq!(got, expected, "row {row}");
    }
    Ok(())
}

fn reference_backward_input(
    x: &[f32],
    grad: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    output_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        let base = row * COLS;
        let xhat = x[base..base + COLS]
            .iter()
            .map(|value| (value - mean[row]) * inv_std[row])
            .collect::<Vec<_>>();
        let sum_grad = grad[base..base + COLS]
            .iter()
            .map(|grad| grad * output_scale)
            .sum::<f32>();
        let sum_xhat_grad = xhat
            .iter()
            .zip(&grad[base..base + COLS])
            .map(|(xhat, grad)| xhat * grad * output_scale)
            .sum::<f32>();
        for col in 0..COLS {
            out[base + col] = (grad[base + col] * output_scale
                - sum_grad / COLS as f32
                - xhat[col] * sum_xhat_grad / COLS as f32)
                * inv_std[row];
        }
    }
    out
}
