use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use rust_kernels_cuda::optimizer::{GradientClipArgs, OptimizerModule};

use super::grads::BackwardBuffers;
use super::next_latent::NextLatGradBuffers;

mod layout;

pub(super) use layout::first_non_finite_gradient;
use layout::{HostGradChunk, gradient_chunks, parameter_gradients};

const GLOBAL_GRAD_CLIP_NORM: f32 = 1.0;

pub(super) struct GradientClipBuffers {
    chunk_ptrs: DeviceBuffer<u64>,
    chunk_lens: DeviceBuffer<u32>,
    chunk_sums: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    chunk_count: u32,
}

pub(super) struct GradientClipResult {
    pub norm: f32,
    pub scale: f32,
}

impl GradientClipBuffers {
    pub(super) fn new(
        stream: &CudaStream,
        grads: &BackwardBuffers,
        next_latent: &NextLatGradBuffers,
    ) -> Result<Self, DriverError> {
        let rows = parameter_gradients(grads, next_latent);
        let chunks = gradient_chunks(&rows);
        let chunk_count = chunks.len() as u32;

        Ok(Self {
            chunk_ptrs: upload(stream, &chunks, |chunk| chunk.ptr)?,
            chunk_lens: upload(stream, &chunks, |chunk| chunk.len)?,
            chunk_sums: DeviceBuffer::zeroed(stream, chunk_count as usize)?,
            scale: DeviceBuffer::zeroed(stream, 1)?,
            norm: DeviceBuffer::zeroed(stream, 1)?,
            chunk_count,
        })
    }

    pub(super) fn clip(
        &mut self,
        stream: &CudaStream,
        optimizer: &OptimizerModule,
    ) -> Result<GradientClipResult, DriverError> {
        optimizer.clip_gradients(GradientClipArgs {
            stream,
            chunk_ptrs: &self.chunk_ptrs,
            chunk_lens: &self.chunk_lens,
            chunk_sums: &mut self.chunk_sums,
            scale: &mut self.scale,
            norm: &mut self.norm,
            chunk_count: self.chunk_count,
            max_norm: GLOBAL_GRAD_CLIP_NORM,
            apply: false,
        })?;
        let scale = self.scale.to_host_vec(stream)?[0];
        let norm = self.norm.to_host_vec(stream)?[0];
        Ok(GradientClipResult { norm, scale })
    }
}

fn upload<T, F>(
    stream: &CudaStream,
    chunks: &[HostGradChunk],
    f: F,
) -> Result<DeviceBuffer<T>, DriverError>
where
    T: DeviceCopy,
    F: Fn(HostGradChunk) -> T,
{
    let values: Vec<T> = chunks.iter().copied().map(f).collect();
    DeviceBuffer::from_host(stream, &values)
}
