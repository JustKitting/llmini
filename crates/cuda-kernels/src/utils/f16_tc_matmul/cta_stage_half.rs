use cuda_device::thread;

use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_THREADS, CtaMatmulDims, CtaTile};

pub(super) fn stage_tiles_half_rhs_lower_a(
    a: &[u16],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    stage_a_lower(a, a_tile, tile, dims, k_base);
    stage_rhs(rhs, b_tile, tile, dims, k_base);
}

pub(super) fn stage_tiles_half_a_transposed_rhs_lower_a(
    a: &[u16],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    stage_a_transposed_lower(a, a_tile, tile, dims, k_base);
    stage_rhs(rhs, b_tile, tile, dims, k_base);
}

fn stage_a_lower(
    src: &[u16],
    dst: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        dst[offset as usize] =
            if global_row < dims.m && global_col < dims.k && global_col <= global_row {
                src[((tile.batch * dims.m + global_row) * dims.k + global_col) as usize]
            } else {
                0
            };
        offset += CTA_THREADS;
    }
}

fn stage_a_transposed_lower(
    src: &[u16],
    dst: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.row_base, k_base);
        dst[offset as usize] =
            if global_row < dims.m && global_col < dims.k && global_col >= global_row {
                src[((tile.batch * dims.k + global_col) * dims.m + global_row) as usize]
            } else {
                0
            };
        offset += CTA_THREADS;
    }
}

fn stage_rhs(
    src: &[u16],
    dst: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(offset, tile.col_base, k_base);
        dst[offset as usize] = if global_row < dims.n && global_col < dims.k {
            src[((tile.batch * dims.k + global_col) * dims.n + global_row) as usize]
        } else {
            0
        };
        offset += CTA_THREADS;
    }
}
