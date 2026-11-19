use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::{fma_f32, max_f32};
use crate::launch::linear_config;
use crate::nvfp4::Nvfp4DeviceTensor;
use crate::nvfp4::nvfp4_value;

const THREADS_PER_BLOCK: u32 = 256;

pub struct ProjectionBiasArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub raw: &'out mut DeviceBuffer<f32>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub rows: u32,
    pub cols: u32,
}

pub struct ProjectionResidualArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub raw: &'scratch DeviceBuffer<f32>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub residual: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
}

pub struct ProjectionRelu2Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub pre_activation: &'out mut DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub rows: u32,
    pub cols: u32,
}

pub struct ProjectionPostOpModule {
    module: module::LoadedModule,
}

impl ProjectionPostOpModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: module::from_module(module)?,
        })
    }

    pub fn bias_inplace(&self, args: ProjectionBiasArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.projection_bias_inplace_kernel(
            args.stream,
            linear_config(args.rows * args.cols, THREADS_PER_BLOCK),
            args.raw,
            args.bias.bytes,
            args.bias.scales,
            args.bias.global_scale,
            args.cols,
        )
    }

    pub fn residual_add(
        &self,
        args: ProjectionResidualArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        self.module.projection_residual_add_kernel(
            args.stream,
            linear_config(args.rows * args.cols, THREADS_PER_BLOCK),
            args.raw,
            args.bias.bytes,
            args.bias.scales,
            args.bias.global_scale,
            args.residual,
            args.cols,
        )
    }

    pub fn relu2_inplace(&self, args: ProjectionRelu2Args<'_, '_>) -> Result<(), DriverError> {
        self.module.projection_relu2_inplace_kernel(
            args.stream,
            linear_config(args.rows * args.cols, THREADS_PER_BLOCK),
            args.pre_activation,
            args.out,
            args.bias.bytes,
            args.bias.scales,
            args.bias.global_scale,
            args.cols,
        )
    }
}

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn projection_bias_inplace_kernel(
        mut raw: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        while index < raw.len() as u32 {
            let col = index - (index / cols) * cols;
            let bias = nvfp4_value(bias_bytes, bias_scales, bias_global_scale[0], col as usize);
            unsafe {
                let slot = raw.get_unchecked_mut(index as usize);
                *slot += bias;
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn projection_residual_add_kernel(
        raw: &[f32],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut residual: DisjointSlice<f32>,
        cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        while index < raw.len() as u32 {
            let col = index - (index / cols) * cols;
            let bias = nvfp4_value(bias_bytes, bias_scales, bias_global_scale[0], col as usize);
            unsafe {
                let slot = residual.get_unchecked_mut(index as usize);
                *slot = fma_f32(1.0, raw[index as usize] + bias, *slot);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn projection_relu2_inplace_kernel(
        mut pre_activation: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        cols: u32,
    ) {
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        while index < pre_activation.len() as u32 {
            let col = index - (index / cols) * cols;
            let bias = nvfp4_value(bias_bytes, bias_scales, bias_global_scale[0], col as usize);
            unsafe {
                let pre_slot = pre_activation.get_unchecked_mut(index as usize);
                let pre = *pre_slot + bias;
                let relu = max_f32(pre, 0.0);
                *pre_slot = pre;
                *out.get_unchecked_mut(index as usize) = crate::activation::centered_relu2_from_positive(relu);
            }
            index += stride;
        }
    }
}
