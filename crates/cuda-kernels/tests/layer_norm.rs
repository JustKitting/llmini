use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, LayerNormArgs, LayerNormModule, ROW_SIZE};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;
#[path = "layer_norm/reference.rs"]
mod reference;

use common::assert_slice_close;
use common::nvfp4::{one_pair_bytes, one_scales};
use reference::{
    assert_row_amax, reference_layer_norm, reference_layer_norm_rows, sample_row_0, sample_row_1,
};

const GPT_EMBEDDING_DIM: usize = 768;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn layer_norm_matches_reference() -> Result<(), Box<dyn Error>> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let x: [f32; ROW_SIZE * 2] = std::array::from_fn(|i| {
        if i < ROW_SIZE {
            sample_row_0(i)
        } else {
            sample_row_1(i - ROW_SIZE)
        }
    });
    let gamma: [f32; ROW_SIZE] = std::array::from_fn(|col| 0.75 + col as f32 * 0.01);
    let beta: [f32; ROW_SIZE] = std::array::from_fn(|col| -0.125 + col as f32 * 0.005);

    let (_, stream, module) = common::cuda_test_module(LayerNormModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let gamma_dev = DeviceBuffer::from_host(&stream, &gamma)?;
    let beta_dev = DeviceBuffer::from_host(&stream, &beta)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;

    module.layer_norm_warp_f32(LayerNormArgs {
        stream: &stream,
        x: &x_dev,
        gamma: &gamma_dev,
        beta: &beta_dev,
        out: &mut out_dev,
        row_count: row_count as u32,
        epsilon,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    let expected = reference_layer_norm(&x, &gamma, &beta, row_count, epsilon);
    assert_slice_close(&out, &expected, 1.0e-9);
    Ok(())
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gpt_layer_norm_matches_reference() -> Result<(), Box<dyn Error>> {
    let row_count = 2usize;
    let epsilon = 1.0e-5f32;
    let output_scale = 0.25f32;
    let x = (0..row_count * GPT_EMBEDDING_DIM)
        .map(|i| {
            ((i % GPT_EMBEDDING_DIM) as f32 - 383.5) * 0.001 + (i / GPT_EMBEDDING_DIM) as f32 * 0.25
        })
        .collect::<Vec<_>>();

    let bias_bytes = vec![0_u8; GPT_EMBEDDING_DIM / 2];

    let (_, stream, module) = common::cuda_test_module(LayerNormModule::from_module)?;

    let x_dev = DeviceBuffer::from_host(&stream, &x)?;
    let weight_bytes_dev = DeviceBuffer::from_host(&stream, &one_pair_bytes(GPT_EMBEDDING_DIM))?;
    let weight_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(GPT_EMBEDDING_DIM))?;
    let bias_bytes_dev = DeviceBuffer::from_host(&stream, &bias_bytes)?;
    let bias_scales_dev = DeviceBuffer::from_host(&stream, &one_scales(GPT_EMBEDDING_DIM))?;
    let weight_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let bias_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, x.len())?;
    let mut amax_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut mean_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;
    let mut inv_std_dev = DeviceBuffer::<f32>::zeroed(&stream, row_count)?;

    module.gpt_layer_norm(GptLayerNormArgs {
        stream: &stream,
        residual: &x_dev,
        weight: Nvfp4DeviceTensor::new(
            &weight_bytes_dev,
            &weight_scales_dev,
            &weight_global_scale_dev,
        ),
        bias: Nvfp4DeviceTensor::new(&bias_bytes_dev, &bias_scales_dev, &bias_global_scale_dev),
        normalized: &mut out_dev,
        normalized_amax: &mut amax_dev,
        mean: &mut mean_dev,
        inv_std: &mut inv_std_dev,
        row_count: row_count as u32,
        embedding_dim: GPT_EMBEDDING_DIM as u32,
        epsilon,
        output_scale,
    })?;

    let out = out_dev.to_host_vec(&stream)?;
    let amax = amax_dev.to_host_vec(&stream)?;
    let expected = reference_layer_norm_rows(&x, row_count, GPT_EMBEDDING_DIM, epsilon)
        .into_iter()
        .map(|value| value * output_scale)
        .collect::<Vec<_>>();
    assert_slice_close(&out, &expected, 1.0e-7);
    assert_row_amax(&out, &amax, row_count, GPT_EMBEDDING_DIM);
    Ok(())
}
