use super::super::super::threads::WARPS_PER_BLOCK;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};
use cuda_device::SharedArray;

#[derive(Clone, Copy)]
pub(super) struct MuonMatrixState {
    pub grad: *const f32,
    pub momentum: *mut f32,
    pub z_master: *mut f32,
    pub x_master: *mut f32,
    pub schedule_amax: *mut f32,
    pub out_fp4: *mut u8,
    pub out_scales: *mut u8,
    pub out_global_scale: *mut f32,
}

#[derive(Clone, Copy)]
pub(super) struct MuonMatrixScratch {
    pub oriented: *mut f32,
    pub polar_next: *mut f32,
    pub polar_x: *mut f32,
    pub polar_gram: *mut f32,
    pub polar_ax: *mut f32,
    pub polar_chunks: *mut f32,
}

pub(super) struct MuonMatrixTiles<'a> {
    pub a_tile: &'a mut SharedArray<u16, CTA_A_ELEMS>,
    pub b_tile: &'a mut SharedArray<u16, CTA_B_ELEMS>,
    pub warp_sums: &'a mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
}

#[derive(Clone, Copy)]
pub(super) struct MuonMatrixShape {
    pub rows: u32,
    pub cols: u32,
}

impl MuonMatrixShape {
    #[inline(always)]
    pub(super) fn len(self) -> u32 {
        self.rows * self.cols
    }
    #[inline(always)]
    pub(super) fn polar_transposed(self) -> bool {
        self.rows > self.cols
    }
    #[inline(always)]
    pub(super) fn master_transposed(self) -> bool {
        self.rows > self.cols
    }
}

#[derive(Clone, Copy)]
pub(super) struct MuonUpdateScalars {
    pub mu: f32,
    pub grad_scale: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
    pub iterations: u32,
}
