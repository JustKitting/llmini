use cuda_device::{SharedArray, convert::cvt_f16x2_f32, thread};

use super::convert::{
    load_f16_global_bits_read_only, load_f32_global_read_only, load_f32x2_global_read_only,
    store_f16x2_shared,
};
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CtaMatmulDims, CtaTile};

macro_rules! stage_tiles_f32_fn {
    ($name:ident, $lhs:ident: $lhs_ty:ty, $rhs:ident: $rhs_ty:ty, $stage_lhs:path, $stage_rhs:path) => {
        pub(super) fn $name(
            $lhs: &[$lhs_ty],
            $rhs: &[$rhs_ty],
            a_tile: &mut super::CtaATile,
            b_tile: &mut super::CtaBTile,
            tile: CtaTile,
            dims: CtaMatmulDims,
            k_base: u32,
        ) {
            $stage_lhs($lhs, a_tile, tile, dims.m, dims.k, k_base);
            $stage_rhs($rhs, b_tile, tile, dims.n, dims.k, k_base);
        }
    };
}

stage_tiles_f32_fn!(stage_tiles_f32_b_t, a: f32, b_t: f32, stage_a, stage_b_t);
stage_tiles_f32_fn!(
    stage_tiles_f32_b_t_aligned,
    a: f32,
    b_t: f32,
    stage_a_aligned,
    stage_b_t_aligned
);
stage_tiles_f32_fn!(
    stage_tiles_f32_rhs_transposed,
    a: f32,
    rhs: f32,
    stage_a,
    stage_rhs_transposed
);
stage_tiles_f32_fn!(
    stage_tiles_f32_half_rhs,
    a: f32,
    rhs: u16,
    stage_a,
    stage_half_rhs_transposed
);

pub(super) fn stage_tiles_f32_half_rhs_lower_a(
    a: &[f32],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    stage_a_lower(a, a_tile, tile, dims.m, dims.k, k_base);
    stage_half_rhs_transposed(rhs, b_tile, tile, dims.n, dims.k, k_base);
}

macro_rules! stage_row_major_f32_fn {
    ($name:ident, $tile_elems:ident, $row_base:ident, $check_bounds:expr) => {
        fn $name(
            src: &[f32],
            dst: &mut SharedArray<u16, $tile_elems>,
            tile: CtaTile,
            rows: u32,
            cols: u32,
            k_base: u32,
        ) {
            stage_row_major_f32::<$check_bounds, $tile_elems>(
                src,
                dst,
                tile,
                tile.$row_base,
                rows,
                cols,
                k_base,
            );
        }
    };
}

stage_row_major_f32_fn!(stage_a, CTA_A_ELEMS, row_base, true);
stage_row_major_f32_fn!(stage_a_aligned, CTA_A_ELEMS, row_base, false);
stage_row_major_f32_fn!(stage_b_t, CTA_B_ELEMS, col_base, true);
stage_row_major_f32_fn!(stage_b_t_aligned, CTA_B_ELEMS, col_base, false);

fn stage_a_lower(
    src: &[f32],
    dst: &mut SharedArray<u16, CTA_A_ELEMS>,
    tile: CtaTile,
    rows: u32,
    cols: u32,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let packed = if global_row < rows && global_col + 1 < cols && global_col + 1 <= global_row {
            let index = ((tile.batch * rows + global_row) * cols + global_col) as usize;
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else if global_row < rows && global_col < cols && global_col <= global_row {
            cvt_f16x2_f32(
                load_f32_global_read_only(
                    src.as_ptr(),
                    ((tile.batch * rows + global_row) * cols + global_col) as usize,
                ),
                0.0,
            )
        } else {
            0
        };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

fn stage_row_major_f32<const CHECK_BOUNDS: bool, const TILE_ELEMS: usize>(
    src: &[f32],
    dst: &mut SharedArray<u16, TILE_ELEMS>,
    tile: CtaTile,
    row_base: u32,
    rows: u32,
    cols: u32,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < TILE_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, row_base, k_base);
        let packed = if !CHECK_BOUNDS || (global_row < rows && global_col + 1 < cols) {
            let index = ((tile.batch * rows + global_row) * cols + global_col) as usize;
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else if global_row < rows && global_col < cols {
            cvt_f16x2_f32(
                load_f32_global_read_only(
                    src.as_ptr(),
                    ((tile.batch * rows + global_row) * cols + global_col) as usize,
                ),
                0.0,
            )
        } else {
            0
        };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

cta_stage_transposed_rhs_fn!(
    stage_rhs_transposed,
    f32,
    load_f32_global_read_only,
    |lo, hi| cvt_f16x2_f32(lo, hi)
);
cta_stage_transposed_rhs_fn!(
    stage_half_rhs_transposed,
    u16,
    load_f16_global_bits_read_only,
    |lo, hi| lo as u32 | ((hi as u32) << 16)
);
