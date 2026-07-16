use cuda_device::{convert::cvt_f16x2_f32, thread};

use super::convert::{load_f32x2_global, store_f16x2_shared};
use super::cta_tile::{CTA_A_ELEMS, CTA_M, CtaMatmulDims, CtaTile};

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
        let local_k = pair / CTA_M;
        let local_row = pair - local_k * CTA_M;
        let global_k = k_base + local_k;
        let global_row = tile.row_base + local_row;
        let packed = if global_k < dims.k && global_row + 1 < dims.m {
            let index = ((tile.batch * dims.k + global_k) * dims.m + global_row) as usize;
            let (lo, hi) = load_f32x2_global(a.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else if global_k < dims.k && global_row < dims.m {
            cvt_f16x2_f32(
                a[((tile.batch * dims.k + global_k) * dims.m + global_row) as usize],
                0.0,
            )
        } else {
            0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
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
        let local_k = pair / CTA_M;
        let local_row = pair - local_k * CTA_M;
        let global_k = k_base + local_k;
        let global_row = tile.row_base + local_row;
        let packed = if global_k < dims.k && global_row + 1 < dims.m && global_k >= global_row + 1 {
            let index = ((tile.batch * dims.k + global_k) * dims.m + global_row) as usize;
            let (lo, hi) = load_f32x2_global(a.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else {
            let lo = if global_k < dims.k && global_row < dims.m && global_k >= global_row {
                a[((tile.batch * dims.k + global_k) * dims.m + global_row) as usize]
            } else {
                0.0
            };
            let hi_row = global_row + 1;
            let hi = if global_k < dims.k && hi_row < dims.m && global_k >= hi_row {
                a[((tile.batch * dims.k + global_k) * dims.m + hi_row) as usize]
            } else {
                0.0
            };
            cvt_f16x2_f32(lo, hi)
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

cta_stage_transposed_rhs_fn!(stage_rhs, f32, |lo, hi| cvt_f16x2_f32(lo, hi));
cta_stage_transposed_rhs_fn!(stage_half_rhs, u16, |lo, hi| lo as u32
    | ((hi as u32) << 16));
