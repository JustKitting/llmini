use cuda_core::DeviceBuffer;
use rust_kernels_cuda::optimizer::GRAD_CLIP_VALUES_PER_CHUNK;

use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;

mod scan;
mod views;

pub(in crate::training) use scan::first_non_finite_gradient;
use views::parameter_gradient_views;

#[derive(Clone, Copy)]
pub(super) struct HostGradPtr {
    pub(super) ptr: u64,
    pub(super) len: u32,
}

#[derive(Clone, Copy)]
pub(super) struct HostGradChunk {
    pub(super) ptr: u64,
    pub(super) len: u32,
}

pub(super) fn parameter_gradients(
    grads: &BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Vec<HostGradPtr> {
    let views = parameter_gradient_views(grads, next_latent);
    let mut rows = Vec::new();
    for view in views {
        push(&mut rows, view.buffer, view.len);
    }
    rows
}

pub(super) fn gradient_chunks(rows: &[HostGradPtr]) -> Vec<HostGradChunk> {
    let mut out = Vec::new();
    for row in rows {
        let mut base = 0u32;
        while base < row.len {
            out.push(HostGradChunk {
                ptr: row.ptr + base as u64 * size_of::<f32>() as u64,
                len: (row.len - base).min(GRAD_CLIP_VALUES_PER_CHUNK as u32),
            });
            base += GRAD_CLIP_VALUES_PER_CHUNK as u32;
        }
    }
    out
}

fn push(rows: &mut Vec<HostGradPtr>, buffer: &DeviceBuffer<f32>, len: usize) {
    rows.push(HostGradPtr {
        ptr: buffer.cu_deviceptr(),
        len: len as u32,
    });
}
