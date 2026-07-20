use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::{
    canon::{CanonBackwardArgs, CanonForwardArgs, CanonModule, CanonQuantizeArgs},
    nvfp4::Nvfp4DeviceTensor,
    nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseDerivedAmaxArgs},
};

mod common;

const BATCHES: usize = 2;
const SEQ_LEN: usize = 5;
const ROWS: usize = BATCHES * SEQ_LEN;
// Exercise the exact hidden width used by the active ~1B model. In particular,
// this covers the full shared-memory row in the fused Canon -> NVFP4 path.
const WIDTH: usize = 2048;
const KERNEL_SIZE: usize = 4;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn canon_forward_and_backward_match_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let module = CanonModule::from_module(ptx.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(ptx)?;

    let residual = (0..ROWS * WIDTH)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) * 0.125)
        .collect::<Vec<_>>();
    let norm_output_scale = 0.5;
    let input = residual
        .iter()
        .map(|value| value * norm_output_scale)
        .collect::<Vec<_>>();
    let weight = (0..WIDTH * KERNEL_SIZE)
        .map(|index| ((index * 5 % 17) as f32 - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let d_output = (0..ROWS * WIDTH)
        .map(|index| ((index * 11 % 29) as f32 - 14.0) * 0.0625)
        .collect::<Vec<_>>();

    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let weight_dev = DeviceBuffer::from_host(&stream, &weight)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    module.forward(CanonForwardArgs {
        stream: &stream,
        input: &input_dev,
        weight: &weight_dev,
        output: &mut output_dev,
        row_count: ROWS as u32,
        seq_len: SEQ_LEN as u32,
        width: WIDTH as u32,
    })?;

    let expected_output = reference_forward(&input, &weight);
    common::assert_slice_close(&output_dev.to_host_vec(&stream)?, &expected_output, 1.0e-6);

    let mut reference_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut reference_fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, input.len() / 2)?;
    let mut reference_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, input.len() / 16)?;
    let mut reference_global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    quant_module.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
        stream: &stream,
        x: &output_dev,
        amax: &mut reference_amax_dev,
        out_fp4: &mut reference_fp4_dev,
        out_scales: &mut reference_scales_dev,
        out_global_scale: &mut reference_global_scale_dev,
        row_count: ROWS as u32,
        row_len: WIDTH as u32,
    })?;

    let mut fused_amax_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    let mut fused_fp4_dev = DeviceBuffer::<u8>::zeroed(&stream, input.len() / 2)?;
    let mut fused_scales_dev = DeviceBuffer::<u8>::zeroed(&stream, input.len() / 16)?;
    let mut fused_global_scale_dev = DeviceBuffer::<f32>::zeroed(&stream, ROWS)?;
    module.forward_quantized(CanonQuantizeArgs {
        stream: &stream,
        input: &input_dev,
        weight: &weight_dev,
        amax: &mut fused_amax_dev,
        out_fp4: &mut fused_fp4_dev,
        out_scales: &mut fused_scales_dev,
        out_global_scale: &mut fused_global_scale_dev,
        row_count: ROWS as u32,
        seq_len: SEQ_LEN as u32,
        width: WIDTH as u32,
    })?;

    common::assert_slice_close(
        &fused_amax_dev.to_host_vec(&stream)?,
        &reference_amax_dev.to_host_vec(&stream)?,
        0.0,
    );
    assert_eq!(
        fused_fp4_dev.to_host_vec(&stream)?,
        reference_fp4_dev.to_host_vec(&stream)?
    );
    assert_eq!(
        fused_scales_dev.to_host_vec(&stream)?,
        reference_scales_dev.to_host_vec(&stream)?
    );
    common::assert_slice_close(
        &fused_global_scale_dev.to_host_vec(&stream)?,
        &reference_global_scale_dev.to_host_vec(&stream)?,
        0.0,
    );

    let residual_f16 = residual
        .iter()
        .copied()
        .map(f32_to_f16_bits)
        .collect::<Vec<_>>();
    let rounded_residual = residual_f16
        .iter()
        .copied()
        .map(f16_bits_to_f32)
        .collect::<Vec<_>>();
    let residual_dev = DeviceBuffer::from_host(&stream, &residual_f16)?;
    let mean_dev = DeviceBuffer::from_host(&stream, &vec![0.0f32; ROWS])?;
    let inv_std_dev = DeviceBuffer::from_host(&stream, &vec![1.0f32; ROWS])?;
    let norm_weight_bytes =
        DeviceBuffer::from_host(&stream, &common::nvfp4::one_pair_bytes(WIDTH))?;
    let norm_weight_scales = DeviceBuffer::from_host(&stream, &common::nvfp4::one_scales(WIDTH))?;
    let norm_bias_bytes = DeviceBuffer::from_host(&stream, &vec![0u8; WIDTH / 2])?;
    let norm_bias_scales = DeviceBuffer::from_host(&stream, &common::nvfp4::one_scales(WIDTH))?;
    let global_scale = DeviceBuffer::from_host(&stream, &[1.0f32])?;
    let d_output_dev = DeviceBuffer::from_host(&stream, &d_output)?;
    let mut d_input_dev = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    let mut d_weight_dev = DeviceBuffer::<f32>::zeroed(&stream, weight.len())?;

    module.backward(CanonBackwardArgs {
        stream: &stream,
        residual_f16: &residual_dev,
        mean: &mean_dev,
        inv_std: &inv_std_dev,
        norm_weight: Nvfp4DeviceTensor::new(&norm_weight_bytes, &norm_weight_scales, &global_scale),
        norm_bias: Nvfp4DeviceTensor::new(&norm_bias_bytes, &norm_bias_scales, &global_scale),
        canon_weight: &weight_dev,
        d_output: &d_output_dev,
        d_input: &mut d_input_dev,
        d_weight: &mut d_weight_dev,
        row_count: ROWS as u32,
        seq_len: SEQ_LEN as u32,
        width: WIDTH as u32,
        norm_output_scale,
    })?;

    let expected_d_input = reference_d_input(&d_output, &weight);
    let expected_d_weight = reference_d_weight(&rounded_residual, &d_output, norm_output_scale);
    common::assert_slice_close(
        &d_input_dev.to_host_vec(&stream)?,
        &expected_d_input,
        1.0e-6,
    );
    common::assert_slice_close(
        &d_weight_dev.to_host_vec(&stream)?,
        &expected_d_weight,
        2.0e-5,
    );
    Ok(())
}

fn reference_forward(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut output = input.to_vec();
    for row in 0..ROWS {
        let position = row % SEQ_LEN;
        for col in 0..WIDTH {
            for lag in 0..KERNEL_SIZE.min(position + 1) {
                output[row * WIDTH + col] +=
                    weight[col * KERNEL_SIZE + lag] * input[(row - lag) * WIDTH + col];
            }
        }
    }
    output
}

fn reference_d_input(d_output: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut d_input = d_output.to_vec();
    for row in 0..ROWS {
        let remaining = SEQ_LEN - row % SEQ_LEN;
        for col in 0..WIDTH {
            for lag in 0..KERNEL_SIZE.min(remaining) {
                d_input[row * WIDTH + col] +=
                    weight[col * KERNEL_SIZE + lag] * d_output[(row + lag) * WIDTH + col];
            }
        }
    }
    d_input
}

fn reference_d_weight(residual: &[f32], d_output: &[f32], output_scale: f32) -> Vec<f32> {
    let mut d_weight = vec![0.0f32; WIDTH * KERNEL_SIZE];
    for row in 0..ROWS {
        let position = row % SEQ_LEN;
        for col in 0..WIDTH {
            for lag in 0..KERNEL_SIZE.min(position + 1) {
                d_weight[col * KERNEL_SIZE + lag] += d_output[row * WIDTH + col]
                    * residual[(row - lag) * WIDTH + col]
                    * output_scale;
            }
        }
    }
    d_weight
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = 14 - exponent;
        let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let rounded = mantissa + 0x0000_0fff + ((mantissa >> 13) & 1);
    if rounded & 0x0080_0000 != 0 {
        let next_exponent = exponent + 1;
        if next_exponent >= 31 {
            sign | 0x7c00
        } else {
            sign | ((next_exponent as u16) << 10)
        }
    } else {
        sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
    }
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
