use cuda_device::{convert::cvt_f16x2_f32, thread};

use super::convert::store_f16x2_shared;
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_THREADS, CtaMatmulDims, CtaTile};

macro_rules! stage_tiles_a_transposed_fn {
    ($name:ident, $rhs:ident: $rhs_ty:ty, $stage_rhs:path) => {
        pub(super) fn $name(
            a: &[f32],
            $rhs: &[$rhs_ty],
            a_tile: &mut super::CtaATile,
            b_tile: &mut super::CtaBTile,
            tile: CtaTile,
            dims: CtaMatmulDims,
            k_base: u32,
        ) {
            stage_a_transposed(a, a_tile, tile, dims, k_base);
            $stage_rhs($rhs, b_tile, tile, dims.n, dims.k, k_base);
        }
    };
}

stage_tiles_a_transposed_fn!(stage_tiles_f32_a_transposed_rhs, rhs: f32, stage_rhs);
stage_tiles_a_transposed_fn!(
    stage_tiles_f32_a_transposed_half_rhs,
    rhs: u16,
    stage_half_rhs
);

pub(super) fn stage_tiles_f32_a_transposed_half_rhs_lower_a(
    a: &[f32],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    stage_a_transposed_lower(a, a_tile, tile, dims, k_base);
    stage_half_rhs(rhs, b_tile, tile, dims.n, dims.k, k_base);
}

fn stage_a_transposed(
    a: &[f32],
    a_tile: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let lo = if global_row < dims.m && global_col < dims.k {
            a[((tile.batch * dims.k + global_col) * dims.m + global_row) as usize]
        } else {
            0.0
        };
        let hi_col = global_col + 1;
        let hi = if global_row < dims.m && hi_col < dims.k {
            a[((tile.batch * dims.k + hi_col) * dims.m + global_row) as usize]
        } else {
            0.0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += CTA_THREADS * 2;
    }
}

fn stage_a_transposed_lower(
    a: &[f32],
    a_tile: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let lo = if global_row < dims.m && global_col < dims.k && global_col >= global_row {
            a[((tile.batch * dims.k + global_col) * dims.m + global_row) as usize]
        } else {
            0.0
        };
        let hi_col = global_col + 1;
        let hi = if global_row < dims.m && hi_col < dims.k && hi_col >= global_row {
            a[((tile.batch * dims.k + hi_col) * dims.m + global_row) as usize]
        } else {
            0.0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += CTA_THREADS * 2;
    }
}

cta_stage_transposed_rhs_fn!(stage_rhs, f32, |lo, hi| cvt_f16x2_f32(lo, hi));
cta_stage_transposed_rhs_fn!(stage_half_rhs, u16, |lo, hi| lo as u32
    | ((hi as u32) << 16));
