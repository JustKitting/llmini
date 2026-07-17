use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::f16_tc_matmul::{F16ConvertArgs, F16TcMatmulModule};
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardModule, LayerNormBackwardParamArgs, LayerNormBackwardParamF32Args,
};

mod common;
#[path = "layer_norm/stats.rs"]
mod stats;

use stats::{reference_row_stats, sample_rows};

const ROWS: usize = 3;
const COLS: usize = 32;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_params_match_reference() -> Result<(), Box<dyn Error>> {
    let epsilon = 1.0e-5f32;
    let output_scale = 0.25f32;
    let x = sample_rows(ROWS, COLS, 19, 9.0, 0.125, 0.25);
    let dy = sample_rows(ROWS, COLS, 13, 6.0, 0.03125, 0.0);
    let (mean, inv_std) = reference_row_stats(&x, ROWS, COLS, epsilon);

    let (_, stream, module) = common::cuda_test_module(LayerNormBackwardModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let mut d_weight_dev = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;
    let mut d_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, COLS)?;

    module.backward_params_f32(LayerNormBackwardParamF32Args {
        stream: &stream,
        residual: &x_dev,
        d_normalized: &dy_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        d_weight: &mut d_weight_dev,
        d_bias: &mut d_bias_dev,
        output_scale,
        row_count: ROWS as u32,
        embedding_dim: COLS as u32,
    })?;

    let (expected_weight, expected_bias) =
        reference_param_grads(&x, &dy, &mean, &inv_std, output_scale);
    common::assert_slice_close(
        &d_weight_dev.to_host_vec(&stream)?,
        &expected_weight,
        1.0e-7,
    );
    common::assert_slice_close(&d_bias_dev.to_host_vec(&stream)?, &expected_bias, 1.0e-7);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_backward_params_tiled_f16_match_reference() -> Result<(), Box<dyn Error>> {
    const TILED_ROWS: usize = 64;
    const TILED_COLS: usize = 32;

    let epsilon = 1.0e-5f32;
    let output_scale = 0.25f32;
    let x = sample_rows(TILED_ROWS, TILED_COLS, 19, 9.0, 0.125, 0.25);
    let dy = sample_rows(TILED_ROWS, TILED_COLS, 13, 6.0, 0.03125, 0.0);
    let (mean, inv_std) = reference_row_stats(&x, TILED_ROWS, TILED_COLS, epsilon);

    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = LayerNormBackwardModule::from_module(ptx.clone())?;
    let f16 = F16TcMatmulModule::from_module(ptx)?;
    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let mut x_f16_dev = DeviceBuffer::<u16>::zeroed(&stream, x.len())?;
    f16.fp32_to_f16(F16ConvertArgs {
        stream: &stream,
        src: &x_dev,
        dst: &mut x_f16_dev,
        element_count: x.len() as u32,
    })?;
    let x_rounded = x_f16_dev
        .to_host_vec(&stream)?
        .into_iter()
        .map(f16_bits_to_f32)
        .collect::<Vec<_>>();
    let dy_dev = DeviceBuffer::from_host(&stream, &dy)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &mean)?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &inv_std)?;
    let mut d_weight_dev = DeviceBuffer::<f32>::zeroed(&stream, TILED_COLS)?;
    let mut d_bias_dev = DeviceBuffer::<f32>::zeroed(&stream, TILED_COLS)?;

    module.backward_params(LayerNormBackwardParamArgs {
        stream: &stream,
        residual: &x_f16_dev,
        d_normalized: &dy_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        d_weight: &mut d_weight_dev,
        d_bias: &mut d_bias_dev,
        output_scale,
        row_count: TILED_ROWS as u32,
        embedding_dim: TILED_COLS as u32,
    })?;

    let (expected_weight, expected_bias) = reference_param_grads_for(
        &x_rounded,
        &dy,
        &mean,
        &inv_std,
        output_scale,
        TILED_ROWS,
        TILED_COLS,
    );
    common::assert_slice_close(
        &d_weight_dev.to_host_vec(&stream)?,
        &expected_weight,
        2.0e-5,
    );
    common::assert_slice_close(&d_bias_dev.to_host_vec(&stream)?, &expected_bias, 2.0e-5);
    Ok(())
}

fn reference_param_grads(
    x: &[f32],
    dy: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    output_scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    reference_param_grads_for(x, dy, mean, inv_std, output_scale, ROWS, COLS)
}

fn reference_param_grads_for(
    x: &[f32],
    dy: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    output_scale: f32,
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_weight = vec![0.0f32; cols];
    let mut d_bias = vec![0.0f32; cols];
    for row in 0..rows {
        for col in 0..cols {
            let offset = row * cols + col;
            let xhat = (x[offset] - mean[row]) * inv_std[row];
            d_weight[col] += dy[offset] * output_scale * xhat;
            d_bias[col] += dy[offset] * output_scale;
        }
    }
    (d_weight, d_bias)
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
