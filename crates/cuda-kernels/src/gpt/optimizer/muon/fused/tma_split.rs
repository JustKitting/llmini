use cuda_device::{
    DisjointSlice, SharedArray, cooperative_launch, cuda_module, grid, kernel, thread,
};

use crate::block_reduce::block_max_store_f32;
use crate::device_ptr::read_f32;
use crate::f16_tc_matmul::convert::{load_f32x2_global, store_f32x2_global};
use crate::float_ptx::{abs_f32, max_f32, sqrt_f32};
use crate::nvfp4_quant::kernels::four_six::helpers::four_six_global_scale;
use crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;
use crate::optimizer::MuonSlotDescriptor;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;
use super::super::polar::fused::{
    normalize_source_to_x, reduce_source_sumsq_chunks_to_inv_norm, source_sumsq_chunks,
};
use super::momentum::momentum_orient;
use super::quant::{encode_four_six, quantize_updated_master};
use super::types::{MuonMatrixShape, MuonUpdateScalars};
use super::update::update_master_chunks;

#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    #[cooperative_launch]
    pub fn muon_tma_prepare_polar_kernel(
        slots: &[MuonSlotDescriptor],
        mut oriented: DisjointSlice<f32>,
        mut polar_x: DisjointSlice<f32>,
        mut polar_chunks: DisjointSlice<f32>,
        slot_index: u32,
        mu: f32,
        grad_scale: f32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let desc = slots[slot_index as usize];
        let shape = MuonMatrixShape {
            rows: desc.rows,
            cols: desc.cols,
        };
        let transposed = shape.polar_transposed();
        let work = WorkGrid::x_axis();

        momentum_orient(
            ptr_const(desc.grad),
            ptr_mut(desc.momentum),
            oriented.as_mut_ptr(),
            work,
            shape,
            mu,
            grad_scale,
            transposed,
        );
        grid::sync();

        let polar_rows = if transposed { desc.cols } else { desc.rows };
        let polar_cols = if transposed { desc.rows } else { desc.cols };
        unsafe {
            normalize_source_to_x(
                oriented.as_mut_ptr(),
                polar_x.as_mut_ptr(),
                polar_chunks.as_mut_ptr(),
                &mut WARP_SUMS,
                work,
                polar_rows,
                polar_cols,
                polar_rows,
                polar_cols,
                false,
            );
        }
    }

    #[kernel]
    pub fn muon_tma_momentum_orient_kernel(
        slots: &[MuonSlotDescriptor],
        mut oriented: DisjointSlice<f32>,
        slot_index: u32,
        mu: f32,
        grad_scale: f32,
    ) {
        let desc = slots[slot_index as usize];
        let shape = MuonMatrixShape {
            rows: desc.rows,
            cols: desc.cols,
        };
        momentum_orient(
            ptr_const(desc.grad),
            ptr_mut(desc.momentum),
            oriented.as_mut_ptr(),
            WorkGrid::x_axis(),
            shape,
            mu,
            grad_scale,
            shape.polar_transposed(),
        );
    }

    #[kernel]
    pub fn muon_tma_source_sumsq_chunks_kernel(
        source: &[f32],
        mut polar_chunks: DisjointSlice<f32>,
        matrix_len: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        unsafe {
            source_sumsq_chunks(
                source.as_ptr(),
                polar_chunks.as_mut_ptr(),
                &mut WARP_SUMS,
                WorkGrid::x_axis(),
                matrix_len,
            );
        }
    }

    #[kernel]
    pub fn muon_tma_reduce_source_norm_kernel(
        mut polar_chunks: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        unsafe {
            reduce_source_sumsq_chunks_to_inv_norm(
                polar_chunks.as_mut_ptr(),
                &mut WARP_SUMS,
                chunk_count,
            );
        }
    }

    #[kernel]
    pub fn muon_tma_scale_source_to_x_kernel(
        source: &[f32],
        mut polar_x: DisjointSlice<f32>,
        polar_chunks: &[f32],
        mut polar_x_chunk_amax: DisjointSlice<f32>,
        matrix_len: u32,
    ) {
        static mut CHUNK_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let inv_norm = polar_chunks[0];
        let chunk = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = thread & (WARP_SIZE - 1);
        let warp = thread / WARP_SIZE;
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let mut offset = thread * 2;
        let mut local_amax = 0.0;
        let out = polar_x.as_mut_ptr();
        while offset < TENSOR_AMAX_VALUES_PER_BLOCK {
            let index = base + offset;
            if index + 1 < matrix_len {
                let i = index as usize;
                let (source0, source1) = load_f32x2_global(source.as_ptr(), i);
                let value0 = source0 * inv_norm;
                let value1 = source1 * inv_norm;
                store_f32x2_global(out, i, value0, value1);
                local_amax = max_f32(local_amax, abs_f32(value0));
                local_amax = max_f32(local_amax, abs_f32(value1));
            } else if index < matrix_len {
                let value = source[index as usize] * inv_norm;
                unsafe { *out.add(index as usize) = value };
                local_amax = max_f32(local_amax, abs_f32(value));
            }
            offset += thread::blockDim_x() * 2;
        }
        block_max_store_f32!(
            CHUNK_AMAX,
            polar_x_chunk_amax[chunk],
            local_amax,
            lane,
            warp
        );
    }

    #[kernel]
    #[cooperative_launch]
    pub fn muon_tma_finish_update_kernel(
        slots: &[MuonSlotDescriptor],
        polar_update: &[f32],
        polar_bound_amax: &[f32],
        mut polar_chunks: DisjointSlice<f32>,
        slot_index: u32,
        learning_rate: f32,
        weight_decay: f32,
        average_coefficient: f32,
        schedule_beta: f32,
        apply_polar_sqrt_bound: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        static mut WARP_MAX_PAIRS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let desc = slots[slot_index as usize];
        let shape = MuonMatrixShape {
            rows: desc.rows,
            cols: desc.cols,
        };
        let len = shape.len();
        let work = WorkGrid::x_axis();
        let scalars = MuonUpdateScalars {
            mu: 0.0,
            grad_scale: 1.0,
            learning_rate: learning_rate * desc.learning_rate_multiplier,
            weight_decay,
            average_coefficient,
            iterations: 0,
        };
        let bound = polar_bound_amax[0];
        let polar_update_scale = if apply_polar_sqrt_bound != 0 && bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };

        unsafe {
            update_master_chunks(
                polar_update.as_ptr(),
                ptr_mut(desc.z_master),
                ptr_mut(desc.x_master),
                ptr_mut(desc.momentum),
                polar_chunks.as_mut_ptr(),
                core::ptr::null(),
                u32::MAX,
                0,
                shape.rows,
                shape.cols,
                len,
                shape.master_transposed(),
                polar_update_scale,
                scalars.learning_rate,
                scalars.weight_decay,
                scalars.average_coefficient,
                schedule_beta,
                &mut WARP_SUMS,
                &mut WARP_MAX_PAIRS,
                work,
            );
        }
        grid::sync();

        unsafe {
            quantize_updated_master(
                ptr_const(desc.x_master),
                polar_chunks.as_mut_ptr(),
                ptr_mut(desc.schedule_amax),
                ptr_mut(desc.bytes),
                ptr_mut(desc.scales),
                ptr_mut(desc.global_scale),
                len,
                &mut WARP_SUMS,
                &mut WARP_MAX_PAIRS,
                work,
            );
        }
    }

    #[kernel]
    pub fn muon_tma_update_master_chunks_kernel(
        slots: &[MuonSlotDescriptor],
        polar_update: &[f32],
        polar_bound_amax: &[f32],
        mut polar_chunks: DisjointSlice<f32>,
        qk_clip_factors: &[f32],
        slot_index: u32,
        learning_rate: f32,
        weight_decay: f32,
        average_coefficient: f32,
        schedule_beta: f32,
        apply_polar_sqrt_bound: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        static mut WARP_MAX_PAIRS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let desc = slots[slot_index as usize];
        let shape = MuonMatrixShape {
            rows: desc.rows,
            cols: desc.cols,
        };
        let bound = polar_bound_amax[0];
        let polar_update_scale = if apply_polar_sqrt_bound != 0 && bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };

        unsafe {
            update_master_chunks(
                polar_update.as_ptr(),
                ptr_mut(desc.z_master),
                ptr_mut(desc.x_master),
                ptr_mut(desc.momentum),
                polar_chunks.as_mut_ptr(),
                qk_clip_factors.as_ptr(),
                desc.qk_clip_factor_offset,
                desc.qk_clip_head_dim,
                shape.rows,
                shape.cols,
                shape.len(),
                shape.master_transposed(),
                polar_update_scale,
                learning_rate * desc.learning_rate_multiplier,
                weight_decay,
                average_coefficient,
                schedule_beta,
                &mut WARP_SUMS,
                &mut WARP_MAX_PAIRS,
                WorkGrid::x_axis(),
            );
        }
    }

    #[kernel]
    pub fn muon_tma_reduce_update_amax_kernel(
        slots: &[MuonSlotDescriptor],
        polar_chunks: &[f32],
        slot_index: u32,
        chunk_count: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        static mut WARP_MAX_PAIRS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let lane = tid & (WARP_SIZE - 1);
        let warp_in_block = tid / WARP_SIZE;
        let chunks = polar_chunks.as_ptr();
        let schedule_chunks = unsafe { chunks.add(chunk_count as usize) };
        let mut chunk = tid;
        let mut local_master_amax = 0.0;
        let mut local_schedule_amax = 0.0;
        while chunk < chunk_count {
            local_master_amax = max_f32(local_master_amax, read_f32(chunks, chunk));
            local_schedule_amax = max_f32(local_schedule_amax, read_f32(schedule_chunks, chunk));
            chunk += thread::blockDim_x();
        }
        if let Some((master_amax, schedule_amax)) = unsafe {
            crate::block_reduce::block_max_pair_leader_f32(
                &mut WARP_SUMS,
                &mut WARP_MAX_PAIRS,
                local_master_amax,
                local_schedule_amax,
                lane,
                warp_in_block,
            )
        } {
            let desc = slots[slot_index as usize];
            unsafe {
                *ptr_mut::<f32>(desc.global_scale) = four_six_global_scale(master_amax, 1.0);
                *ptr_mut::<f32>(desc.schedule_amax) = schedule_amax;
            }
        }
    }

    #[kernel]
    pub fn muon_tma_encode_updated_master_kernel(slots: &[MuonSlotDescriptor], slot_index: u32) {
        let desc = slots[slot_index as usize];
        let shape = MuonMatrixShape {
            rows: desc.rows,
            cols: desc.cols,
        };
        encode_four_six(
            ptr_const(desc.x_master),
            ptr_mut(desc.bytes),
            ptr_mut(desc.scales),
            ptr_mut(desc.global_scale),
            shape.len(),
            WorkGrid::x_axis(),
        );
    }
}

#[inline(always)]
fn ptr_const<T>(ptr: u64) -> *const T {
    ptr as usize as *const T
}

#[inline(always)]
fn ptr_mut<T>(ptr: u64) -> *mut T {
    ptr as usize as *mut T
}
