use cuda_device::thread;

use super::convert::{load_f16x2_global_bits, store_f16x2_shared};
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_M, CTA_N, CtaMatmulDims, CtaTile};

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
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let packed =
            if global_row < dims.m && global_col + 1 < dims.k && global_col + 1 <= global_row {
                let index = ((tile.batch * dims.m + global_row) * dims.k + global_col) as usize;
                load_f16x2_global_bits(src.as_ptr(), index)
            } else if global_row < dims.m && global_col < dims.k && global_col <= global_row {
                src[((tile.batch * dims.m + global_row) * dims.k + global_col) as usize] as u32
            } else {
                0
            };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

fn stage_a_transposed_lower(
    src: &[u16],
    dst: &mut super::CtaATile,
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
            load_f16x2_global_bits(src.as_ptr(), index)
        } else {
            let lo = if global_k < dims.k && global_row < dims.m && global_k >= global_row {
                src[((tile.batch * dims.k + global_k) * dims.m + global_row) as usize]
            } else {
                0
            };
            let hi_row = global_row + 1;
            let hi = if global_k < dims.k && hi_row < dims.m && global_k >= hi_row {
                src[((tile.batch * dims.k + global_k) * dims.m + hi_row) as usize]
            } else {
                0
            };
            lo as u32 | ((hi as u32) << 16)
        };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

fn stage_rhs(
    src: &[u16],
    dst: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let local_k = pair / CTA_N;
        let local_col = pair - local_k * CTA_N;
        let global_k = k_base + local_k;
        let global_col = tile.col_base + local_col;
        let packed = if global_k < dims.k && global_col + 1 < dims.n {
            let index = ((tile.batch * dims.k + global_k) * dims.n + global_col) as usize;
            load_f16x2_global_bits(src.as_ptr(), index)
        } else if global_k < dims.k && global_col < dims.n {
            src[((tile.batch * dims.k + global_k) * dims.n + global_col) as usize] as u32
        } else {
            0
        };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}
