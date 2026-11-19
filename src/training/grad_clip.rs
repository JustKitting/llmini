use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use std::collections::{HashMap, HashSet};

use rust_kernels_cuda::optimizer::{GradientClipArgs, OptimizerModule};

use super::grads::BackwardBuffers;
use super::next_latent::NextLatGradBuffers;
use super::optimizer_muon::MuonPointerTables;

mod components;
mod layout;

pub(super) use layout::first_non_finite_gradient;
use layout::{HostGradChunk, gradient_chunks, parameter_gradients};

const GLOBAL_GRAD_CLIP_NORM: f32 = 1.0;

pub(super) struct GradientClipBuffers {
    recomputed_matrix_ptrs: HashSet<u64>,
    recomputed_matrix_sums: Vec<(u32, u32)>,
    extra_gradients: Vec<(String, u64, usize)>,
    chunk_ptrs: DeviceBuffer<u64>,
    chunk_lens: DeviceBuffer<u32>,
    chunk_sums: DeviceBuffer<f32>,
    pub(super) matrix_sums: DeviceBuffer<f32>,
    pub(super) matrix_sum_offsets: Vec<u32>,
    scale: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    chunk_count: u32,
    matrix_sum_count: u32,
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
        muon_tables: &MuonPointerTables,
    ) -> Result<Self, DriverError> {
        Self::new_with_extra(stream, grads, next_latent, muon_tables, &[])
    }

    pub(super) fn new_with_extra(
        stream: &CudaStream,
        grads: &BackwardBuffers,
        next_latent: &NextLatGradBuffers,
        muon_tables: &MuonPointerTables,
        extra: &[(&str, &DeviceBuffer<f32>)],
    ) -> Result<Self, DriverError> {
        Self::new_with_recomputed_matrices(stream, grads, next_latent, muon_tables, extra, &[])
    }

    // A matrix modified after its GEMM (for example structural padding masked
    // after stochastic quantization) must be reduced from its final gradient.
    // Keep the SEL output addresses allocated, but exclude those cached sums.
    pub(super) fn new_with_recomputed_matrices(
        stream: &CudaStream,
        grads: &BackwardBuffers,
        next_latent: &NextLatGradBuffers,
        muon_tables: &MuonPointerTables,
        extra: &[(&str, &DeviceBuffer<f32>)],
        recompute: &[u64],
    ) -> Result<Self, DriverError> {
        let recomputed_matrix_ptrs: HashSet<u64> = recompute.iter().copied().collect();
        let mut recomputed_matrix_sums = Vec::new();
        let mut rows = parameter_gradients(grads, next_latent);
        for (_, buffer) in extra {
            rows.push(layout::HostGradPtr {
                ptr: buffer.cu_deviceptr(),
                len: buffer.len() as u32,
            });
        }
        let mut matrix_sum_offsets = vec![0; muon_tables.slot_count];
        let mut matrix_sum_count = 0;
        if super::symexp_lin::learn_scales() {
            let matrix_grad_ranges = muon_tables.matrix_grad_ranges();
            let matrix_ranges: HashMap<_, _> = matrix_grad_ranges.iter().copied().collect();
            assert_eq!(
                matrix_ranges.len(),
                matrix_grad_ranges.len(),
                "Muon matrix gradient pointers must be unique",
            );
            let mut matched_matrix_rows = 0;
            rows.retain_mut(|row| {
                let Some(&matrix_len) = matrix_ranges.get(&row.ptr) else {
                    return true;
                };
                matched_matrix_rows += 1;
                if recomputed_matrix_ptrs.contains(&row.ptr) {
                    return true;
                }
                assert!(
                    matrix_len <= row.len,
                    "Muon matrix gradient must fit its global-norm row",
                );
                row.ptr += matrix_len as u64 * size_of::<f32>() as u64;
                row.len -= matrix_len;
                row.len > 0
            });
            assert_eq!(
                matched_matrix_rows,
                matrix_ranges.len(),
                "each Muon matrix gradient must match one global-norm row",
            );

            for (slot_index, desc) in muon_tables.all.host_symexp_lin_slots.iter().enumerate() {
                matrix_sum_offsets[slot_index] = matrix_sum_count;
                if desc.rows > 0 && desc.cols > 0 {
                    // The learned SEL controls are parameters too. Their
                    // reductions can be nonfinite even when the transformed
                    // matrix gradient is finite; include them before any
                    // optimizer state advances.
                    for ptr in [desc.exponential_grad, desc.linear_grad, desc.curvature_grad] {
                        if ptr != 0 && !rows.iter().any(|row| row.ptr == ptr) {
                            rows.push(layout::HostGradPtr { ptr, len: 1 });
                        }
                    }
                    use rust_kernels_cuda::nvfp4_tma_matmul::kernels::{TILE_M, TILE_N};
                    let count = desc.cols.div_ceil(TILE_M) * desc.rows.div_ceil(TILE_N);
                    if recomputed_matrix_ptrs.contains(&desc.grad) {
                        recomputed_matrix_sums.push((matrix_sum_count, count));
                    }
                    matrix_sum_count += count;
                }
            }
        }
        let chunks = gradient_chunks(&rows);
        let chunk_count = chunks.len() as u32;

        Ok(Self {
            recomputed_matrix_ptrs,
            recomputed_matrix_sums,
            extra_gradients: extra
                .iter()
                .map(|(name, b)| (name.to_string(), b.cu_deviceptr(), b.len()))
                .collect(),
            chunk_ptrs: upload(stream, &chunks, |chunk| chunk.ptr)?,
            chunk_lens: upload(stream, &chunks, |chunk| chunk.len)?,
            chunk_sums: DeviceBuffer::zeroed(stream, chunk_count as usize)?,
            matrix_sums: DeviceBuffer::zeroed(stream, matrix_sum_count.max(1) as usize)?,
            matrix_sum_offsets,
            scale: DeviceBuffer::zeroed(stream, 1)?,
            norm: DeviceBuffer::zeroed(stream, 1)?,
            chunk_count,
            matrix_sum_count,
        })
    }

    pub(super) fn clip(
        &mut self,
        stream: &CudaStream,
        optimizer: &OptimizerModule,
    ) -> Result<GradientClipResult, DriverError> {
        for &(start, count) in &self.recomputed_matrix_sums {
            unsafe {
                cuda_core::memory::memset_d8_async(
                    self.matrix_sums.cu_deviceptr() + u64::from(start) * 4,
                    0,
                    count as usize * 4,
                    stream.cu_stream(),
                )?;
            }
        }
        optimizer.clip_gradients(GradientClipArgs {
            stream,
            chunk_ptrs: &self.chunk_ptrs,
            chunk_lens: &self.chunk_lens,
            chunk_sums: &mut self.chunk_sums,
            extra_sums: &self.matrix_sums,
            scale: &mut self.scale,
            norm: &mut self.norm,
            chunk_count: self.chunk_count,
            extra_sum_count: self.matrix_sum_count,
            max_norm: GLOBAL_GRAD_CLIP_NORM,
            apply: false,
        })?;
        let scale = self.scale.to_host_vec(stream)?[0];
        let norm = self.norm.to_host_vec(stream)?[0];
        Ok(GradientClipResult { norm, scale })
    }

    pub(super) fn first_non_finite_extra(
        &self,
        stream: &CudaStream,
    ) -> Result<Option<(String, usize, f32)>, DriverError> {
        for (name, ptr, len) in &self.extra_gradients {
            let mut host = vec![0.0f32; *len];
            unsafe {
                cuda_core::memory::memcpy_dtoh_async(
                    host.as_mut_ptr(),
                    *ptr,
                    *len * size_of::<f32>(),
                    stream.cu_stream(),
                )?;
            }
            stream.synchronize()?;
            if let Some((index, value)) = host.into_iter().enumerate().find(|(_, v)| !v.is_finite())
            {
                return Ok(Some((name.clone(), index, value)));
            }
        }
        Ok(None)
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
    let uploaded = DeviceBuffer::from_host(stream, &values);
    // The CUDA upload borrows this temporary host vector until stream completion.
    let completion = stream.synchronize();
    let uploaded = uploaded?;
    completion?;
    Ok(uploaded)
}
