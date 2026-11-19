//! Normalization at the exit of an H-Net stack, before any channel truncation.
use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::{
    block_reduce::block_sum_shared_f32,
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::{fma_f32, sqrt_f32},
    launch::{grid_x_config, launch_config, linear_config},
    nvfp4::{Nvfp4DeviceTensor, nvfp4_vector_value},
    warp_reduce::thread_lane_warp,
};

const THREADS: u32 = 128;
const WARPS: usize = (THREADS / 32) as usize;
const ROWS_PER_PART: u32 = 256;
const EPSILON: f32 = 1.0e-5;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RmsShape {
    pub batch: u32,
    pub stride: u32,
    pub width: u32,
    pub output_width: u32,
}

unsafe impl DeviceCopy for RmsShape {}

pub struct RmsForward<'a, 'scale> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<f32>,
    pub lengths: &'a DeviceBuffer<u32>,
    pub scale: Nvfp4DeviceTensor<'scale>,
    pub shape: RmsShape,
}

pub struct RmsTape<'a> {
    shape: RmsShape,
    scale: Nvfp4DeviceTensor<'a>,
    xhat: DeviceBuffer<u16>,
    inverse: DeviceBuffer<f32>,
}

pub struct RmsNormModule(kernels::LoadedModule);

impl RmsShape {
    fn rows(self) -> Result<u32, DriverError> {
        let rows = self.batch.checked_mul(self.stride).ok_or(DriverError(1))?;
        if rows == 0
            || self.output_width == 0
            || self.output_width > self.width
            || !self.width.is_multiple_of(16)
            || rows.checked_mul(self.width).is_none()
        {
            return Err(DriverError(1));
        }
        Ok(rows)
    }
}

impl<'a> RmsTape<'a> {
    fn new(args: &RmsForward<'_, 'a>, rows: u32) -> Result<Self, DriverError> {
        let shape = args.shape;
        let expected = (rows * shape.width) as usize;
        if args.residual.len() < expected
            || args.lengths.len() != shape.batch as usize
            || args.scale.bytes.len() != shape.width as usize / 2
            || args.scale.scales.len() < crate::nvfp4::csf::encoded_len(1, shape.width as usize)
            || args.scale.global_scale.len() != 1
        {
            return Err(DriverError(1));
        }
        Ok(Self {
            shape,
            scale: args.scale,
            xhat: DeviceBuffer::zeroed(args.stream, expected)?,
            inverse: DeviceBuffer::zeroed(args.stream, rows as usize)?,
        })
    }
}

impl RmsNormModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self(kernels::from_module(module)?))
    }

    pub fn residual<'a>(
        &self,
        args: RmsForward<'_, 'a>,
    ) -> Result<(DeviceBuffer<f32>, RmsTape<'a>), DriverError> {
        let rows = args.shape.rows()?;
        let mut tape = RmsTape::new(&args, rows)?;
        let mut output =
            DeviceBuffer::zeroed(args.stream, (rows * args.shape.output_width) as usize)?;
        self.0.hnet_rms_residual_kernel(
            args.stream,
            grid_x_config(rows, THREADS),
            args.residual,
            args.lengths,
            args.scale.bytes,
            args.scale.scales,
            args.scale.global_scale,
            &mut tape.xhat,
            &mut tape.inverse,
            &mut output,
            args.shape,
        )?;
        Ok((output, tape))
    }

    pub fn represented<'a>(
        &self,
        args: RmsForward<'_, 'a>,
    ) -> Result<(DeviceBuffer<u16>, RmsTape<'a>), DriverError> {
        let rows = args.shape.rows()?;
        let mut tape = RmsTape::new(&args, rows)?;
        let mut output =
            DeviceBuffer::zeroed(args.stream, (rows * args.shape.output_width) as usize)?;
        self.0.hnet_rms_represented_kernel(
            args.stream,
            grid_x_config(rows, THREADS),
            args.residual,
            args.lengths,
            args.scale.bytes,
            args.scale.scales,
            args.scale.global_scale,
            &mut tape.xhat,
            &mut tape.inverse,
            &mut output,
            args.shape,
        )?;
        Ok((output, tape))
    }

    pub fn backward(
        &self,
        stream: &CudaStream,
        tape: RmsTape<'_>,
        gradient: &DeviceBuffer<f32>,
    ) -> Result<(DeviceBuffer<f32>, DeviceBuffer<f32>), DriverError> {
        let shape = tape.shape;
        let rows = shape.rows()?;
        if gradient.len() != (rows * shape.output_width) as usize {
            return Err(DriverError(1));
        }
        let mut input = DeviceBuffer::zeroed(stream, (rows * shape.width) as usize)?;
        let mut scale = DeviceBuffer::zeroed(stream, shape.width as usize)?;
        self.0.hnet_rms_input_gradient_kernel(
            stream,
            grid_x_config(rows, THREADS),
            &tape.xhat,
            &tape.inverse,
            tape.scale.bytes,
            tape.scale.scales,
            tape.scale.global_scale,
            gradient,
            &mut input,
            shape,
        )?;
        let parts = rows.div_ceil(ROWS_PER_PART);
        let mut partials = DeviceBuffer::zeroed(stream, (parts * shape.output_width) as usize)?;
        self.0.hnet_rms_scale_partials_kernel(
            stream,
            launch_config((shape.output_width.div_ceil(THREADS), parts, 1), THREADS),
            &tape.xhat,
            &tape.inverse,
            gradient,
            &mut partials,
            rows,
            shape,
        )?;
        self.0.hnet_rms_scale_sum_kernel(
            stream,
            linear_config(shape.output_width, THREADS),
            &partials,
            &mut scale,
            parts,
            shape.output_width,
        )?;
        Ok((input, scale))
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward(
        input: &[f32],
        lengths: &[u32],
        weights: &[u8],
        scales: &[u8],
        global: f32,
        mut xhat: DisjointSlice<u16>,
        mut inverse: DisjointSlice<f32>,
        shape: RmsShape,
        reduction: &mut SharedArray<f32, WARPS>,
        mut store: impl FnMut(usize, f32),
    ) {
        let row = thread::blockIdx_x();
        let (tid, lane, warp) = thread_lane_warp();
        let full_base = (row * shape.width) as usize;
        let out_base = (row * shape.output_width) as usize;
        let valid = row % shape.stride < lengths[(row / shape.stride) as usize];
        let inv = if valid {
            let mut square = 0.0;
            for col in (tid..shape.width).step_by(THREADS as usize) {
                let value = input[full_base + col as usize];
                square = fma_f32(value, value, square);
            }
            let square = block_sum_shared_f32(reduction, square, lane, warp);
            1.0 / sqrt_f32(square / shape.width as f32 + EPSILON)
        } else {
            0.0
        };
        if tid == 0 {
            unsafe { *inverse.get_unchecked_mut(row as usize) = inv };
        }
        for col in (tid..shape.width).step_by(THREADS as usize) {
            let col = col as usize;
            let normalized = if valid {
                input[full_base + col] * inv
            } else {
                0.0
            };
            unsafe { *xhat.get_unchecked_mut(full_base + col) = cvt_rn_f16_f32(normalized) };
            if col < shape.output_width as usize {
                let value = if valid {
                    nvfp4_vector_value(weights, scales, global, col) * normalized
                } else {
                    0.0
                };
                store(out_base + col, value);
            }
        }
    }

    #[kernel]
    pub fn hnet_rms_residual_kernel(
        input: &[f32],
        lengths: &[u32],
        weights: &[u8],
        scales: &[u8],
        global: &[f32],
        xhat: DisjointSlice<u16>,
        inverse: DisjointSlice<f32>,
        mut output: DisjointSlice<f32>,
        shape: RmsShape,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        forward(
            input,
            lengths,
            weights,
            scales,
            global[0],
            xhat,
            inverse,
            shape,
            unsafe { &mut REDUCTION },
            |at, value| unsafe { *output.get_unchecked_mut(at) = value },
        );
    }

    #[kernel]
    pub fn hnet_rms_represented_kernel(
        input: &[f32],
        lengths: &[u32],
        weights: &[u8],
        scales: &[u8],
        global: &[f32],
        xhat: DisjointSlice<u16>,
        inverse: DisjointSlice<f32>,
        mut output: DisjointSlice<u16>,
        shape: RmsShape,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        forward(
            input,
            lengths,
            weights,
            scales,
            global[0],
            xhat,
            inverse,
            shape,
            unsafe { &mut REDUCTION },
            |at, value| unsafe { *output.get_unchecked_mut(at) = cvt_rn_f16_f32(value) },
        );
    }

    #[kernel]
    pub fn hnet_rms_input_gradient_kernel(
        xhat: &[u16],
        inverse: &[f32],
        weights: &[u8],
        scales: &[u8],
        global: &[f32],
        gradient: &[f32],
        mut output: DisjointSlice<f32>,
        shape: RmsShape,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let (tid, lane, warp) = thread_lane_warp();
        let inv = inverse[row as usize];
        let full = (row * shape.width) as usize;
        let narrow = (row * shape.output_width) as usize;
        if inv == 0.0 {
            for col in (tid..shape.width).step_by(THREADS as usize) {
                unsafe { *output.get_unchecked_mut(full + col as usize) = 0.0 };
            }
            return;
        }
        let scaled = |col: usize| {
            gradient[narrow + col] * nvfp4_vector_value(weights, scales, global[0], col)
        };
        let mut dot = 0.0;
        for col in (tid..shape.output_width).step_by(THREADS as usize) {
            let col = col as usize;
            dot = fma_f32(scaled(col), cvt_f32_f16(xhat[full + col]), dot);
        }
        let mean =
            unsafe { block_sum_shared_f32(&mut REDUCTION, dot, lane, warp) } / shape.width as f32;
        for col in (tid..shape.width).step_by(THREADS as usize) {
            let col = col as usize;
            let direct = if col < shape.output_width as usize {
                scaled(col)
            } else {
                0.0
            };
            let value = inv * (direct - cvt_f32_f16(xhat[full + col]) * mean);
            unsafe { *output.get_unchecked_mut(full + col) = value };
        }
    }

    #[kernel]
    pub fn hnet_rms_scale_partials_kernel(
        xhat: &[u16],
        inverse: &[f32],
        gradient: &[f32],
        mut partials: DisjointSlice<f32>,
        rows: u32,
        shape: RmsShape,
    ) {
        let part = thread::blockIdx_y();
        let column = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if column < shape.output_width {
            let first = part * ROWS_PER_PART;
            let end = (first + ROWS_PER_PART).min(rows);
            let mut sum = 0.0;
            for row in first..end {
                if inverse[row as usize] != 0.0 {
                    let normalized = cvt_f32_f16(xhat[(row * shape.width + column) as usize]);
                    sum = fma_f32(
                        normalized,
                        gradient[(row * shape.output_width + column) as usize],
                        sum,
                    );
                }
            }
            unsafe {
                *partials.get_unchecked_mut((part * shape.output_width + column) as usize) = sum
            };
        }
    }

    #[kernel]
    pub fn hnet_rms_scale_sum_kernel(
        partials: &[f32],
        mut scale: DisjointSlice<f32>,
        parts: u32,
        width: u32,
    ) {
        let column = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if column < width {
            let mut sum = 0.0;
            for part in 0..parts {
                sum += partials[(part * width + column) as usize];
            }
            unsafe { *scale.get_unchecked_mut(column as usize) = sum };
        }
    }
}
