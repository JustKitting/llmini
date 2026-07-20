use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::f16_tc_matmul::{F16ConvertArgs, F16TcMatmulModule};

pub fn saved_f16(
    stream: &CudaStream,
    module: &F16TcMatmulModule,
    values: &[f32],
) -> Result<(DeviceBuffer<u16>, Vec<f32>), Box<dyn Error>> {
    let src = DeviceBuffer::from_host(stream, values)?;
    let mut dst = DeviceBuffer::<u16>::zeroed(stream, values.len())?;
    module.fp32_to_f16(F16ConvertArgs {
        stream,
        src: &src,
        dst: &mut dst,
        element_count: values.len() as u32,
    })?;
    let rounded = dst
        .to_host_vec(stream)?
        .into_iter()
        .map(f16_bits_to_f32)
        .collect();
    Ok((dst, rounded))
}

pub fn f16_bits_to_f32(bits: u16) -> f32 {
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
