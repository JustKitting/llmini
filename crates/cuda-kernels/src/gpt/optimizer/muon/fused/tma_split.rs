use cuda_device::{DisjointSlice, SharedArray, cooperative_launch, cuda_module, grid, kernel};

use crate::float_ptx::sqrt_f32;
use crate::optimizer::MuonSlotDescriptor;

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;
use super::super::polar::fused::normalize_source_to_x;
use super::momentum::momentum_orient;
use super::quant::quantize_updated_master;
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
        apply_polar_sqrt_bound: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

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
                polar_chunks.as_mut_ptr(),
                shape.rows,
                shape.cols,
                len,
                shape.master_transposed(),
                polar_update_scale,
                scalars.learning_rate,
                scalars.weight_decay,
                scalars.average_coefficient,
                &mut WARP_SUMS,
                work,
            );
        }
        grid::sync();

        unsafe {
            quantize_updated_master(
                ptr_const(desc.x_master),
                polar_chunks.as_mut_ptr(),
                ptr_mut(desc.bytes),
                ptr_mut(desc.scales),
                ptr_mut(desc.global_scale),
                len,
                &mut WARP_SUMS,
                work,
            );
        }
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
