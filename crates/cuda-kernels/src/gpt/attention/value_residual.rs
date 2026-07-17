use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::AttentionModule;
use crate::launch::linear_config;

const THREADS_PER_BLOCK: u32 = 256;
const CURRENT_VALUE_WEIGHT: f32 = 0.5;
const FIRST_VALUE_WEIGHT: f32 = 0.5;

pub struct CaptureValueResidualArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub first_value: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
}

pub struct MixValueResidualArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'out mut DeviceBuffer<f32>,
    pub first_value: &'a DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
}

pub struct InitializeValueResidualGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_first_value: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
}

pub struct AccumulateValueResidualGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_first_value: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
}

pub struct FinishValueResidualGradArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_first_value: &'a DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
}

impl AttentionModule {
    pub fn capture_value_residual(
        &self,
        args: CaptureValueResidualArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.value_residual.capture_value_residual_kernel(
            args.stream,
            value_config(args.row_count, args.embedding_dim),
            args.qkv,
            args.first_value,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
        )
    }

    pub fn mix_value_residual(
        &self,
        args: MixValueResidualArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.value_residual.mix_value_residual_kernel(
            args.stream,
            value_config(args.row_count, args.embedding_dim),
            args.qkv,
            args.first_value,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
        )
    }

    pub fn initialize_value_residual_grad(
        &self,
        args: InitializeValueResidualGradArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.value_residual.initialize_value_residual_grad_kernel(
            args.stream,
            value_config(args.row_count, args.embedding_dim),
            args.d_qkv,
            args.d_first_value,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
        )
    }

    pub fn accumulate_value_residual_grad(
        &self,
        args: AccumulateValueResidualGradArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.value_residual.accumulate_value_residual_grad_kernel(
            args.stream,
            value_config(args.row_count, args.embedding_dim),
            args.d_qkv,
            args.d_first_value,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
        )
    }

    pub fn finish_value_residual_grad(
        &self,
        args: FinishValueResidualGradArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.value_residual.finish_value_residual_grad_kernel(
            args.stream,
            value_config(args.row_count, args.embedding_dim),
            args.d_qkv,
            args.d_first_value,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
        )
    }
}

fn value_config(row_count: u32, embedding_dim: u32) -> cuda_core::LaunchConfig {
    linear_config(row_count * embedding_dim, THREADS_PER_BLOCK)
}

#[cuda_module]
pub(super) mod kernels {
    use super::*;

    #[kernel]
    pub fn capture_value_residual_kernel(
        qkv: &[f32],
        mut first_value: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
    ) {
        if let Some((value_index, qkv_index)) = value_position(row_count, embedding_dim, qkv_dim) {
            unsafe {
                *first_value.get_unchecked_mut(value_index) = qkv[qkv_index];
            }
        }
    }

    #[kernel]
    pub fn mix_value_residual_kernel(
        mut qkv: DisjointSlice<f32>,
        first_value: &[f32],
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
    ) {
        if let Some((value_index, qkv_index)) = value_position(row_count, embedding_dim, qkv_dim) {
            unsafe {
                let current = qkv.get_unchecked_mut(qkv_index);
                *current =
                    CURRENT_VALUE_WEIGHT * *current + FIRST_VALUE_WEIGHT * first_value[value_index];
            }
        }
    }

    #[kernel]
    pub fn initialize_value_residual_grad_kernel(
        mut d_qkv: DisjointSlice<f32>,
        mut d_first_value: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
    ) {
        if let Some((value_index, qkv_index)) = value_position(row_count, embedding_dim, qkv_dim) {
            unsafe {
                let current = d_qkv.get_unchecked_mut(qkv_index);
                let weighted = CURRENT_VALUE_WEIGHT * *current;
                *current = weighted;
                *d_first_value.get_unchecked_mut(value_index) = weighted;
            }
        }
    }

    #[kernel]
    pub fn accumulate_value_residual_grad_kernel(
        mut d_qkv: DisjointSlice<f32>,
        mut d_first_value: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
    ) {
        if let Some((value_index, qkv_index)) = value_position(row_count, embedding_dim, qkv_dim) {
            unsafe {
                let current = d_qkv.get_unchecked_mut(qkv_index);
                let weighted = CURRENT_VALUE_WEIGHT * *current;
                *current = weighted;
                *d_first_value.get_unchecked_mut(value_index) += weighted;
            }
        }
    }

    #[kernel]
    pub fn finish_value_residual_grad_kernel(
        mut d_qkv: DisjointSlice<f32>,
        d_first_value: &[f32],
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
    ) {
        if let Some((value_index, qkv_index)) = value_position(row_count, embedding_dim, qkv_dim) {
            unsafe {
                *d_qkv.get_unchecked_mut(qkv_index) += d_first_value[value_index];
            }
        }
    }

    #[inline(always)]
    fn value_position(row_count: u32, embedding_dim: u32, qkv_dim: u32) -> Option<(usize, usize)> {
        let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = row_count * embedding_dim;
        if index >= len {
            return None;
        }
        let row = index / embedding_dim;
        let col = index % embedding_dim;
        let qkv_index = row * qkv_dim + 2 * embedding_dim + col;
        Some((index as usize, qkv_index as usize))
    }
}
