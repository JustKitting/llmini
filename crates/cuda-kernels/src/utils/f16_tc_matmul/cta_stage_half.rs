use cuda_device::thread;

use super::convert::{
    load_f16_global_bits_read_only, load_f16x2_global_bits_read_only, store_f16x2_shared,
};
use super::cta_stage::stage_coords;
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CtaMatmulDims, CtaTile};

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

pub(super) fn stage_tiles_half_rhs_windowed_lower_a(
    a: &[u16],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
    window: u32,
) {
    stage_a_windowed_lower(a, a_tile, tile, dims, k_base, window);
    stage_rhs(rhs, b_tile, tile, dims, k_base);
}

pub(super) fn stage_tiles_half_a_transposed_rhs_windowed_lower_a(
    a: &[u16],
    rhs: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
    window: u32,
) {
    stage_a_transposed_windowed_lower(a, a_tile, tile, dims, k_base, window);
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
                load_f16x2_global_bits_read_only(src.as_ptr(), index)
            } else if global_row < dims.m && global_col < dims.k && global_col <= global_row {
                load_f16_global_bits_read_only(
                    src.as_ptr(),
                    ((tile.batch * dims.m + global_row) * dims.k + global_col) as usize,
                ) as u32
            } else {
                0
            };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

fn stage_a_windowed_lower(
    src: &[u16],
    dst: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
    window: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let packed = if global_row < dims.m
            && global_col + 1 < dims.k
            && in_causal_window(global_row, global_col, window)
            && in_causal_window(global_row, global_col + 1, window)
        {
            let index = ((tile.batch * dims.m + global_row) * dims.k + global_col) as usize;
            load_f16x2_global_bits_read_only(src.as_ptr(), index)
        } else {
            let lo = if global_row < dims.m
                && global_col < dims.k
                && in_causal_window(global_row, global_col, window)
            {
                load_f16_global_bits_read_only(
                    src.as_ptr(),
                    ((tile.batch * dims.m + global_row) * dims.k + global_col) as usize,
                )
            } else {
                0
            };
            let hi_col = global_col + 1;
            let hi = if global_row < dims.m
                && hi_col < dims.k
                && in_causal_window(global_row, hi_col, window)
            {
                load_f16_global_bits_read_only(
                    src.as_ptr(),
                    ((tile.batch * dims.m + global_row) * dims.k + hi_col) as usize,
                )
            } else {
                0
            };
            lo as u32 | ((hi as u32) << 16)
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
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let lo = if global_row < dims.m && global_col < dims.k && global_col >= global_row {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + global_col) * dims.m + global_row) as usize,
            )
        } else {
            0
        };
        let hi_col = global_col + 1;
        let hi = if global_row < dims.m && hi_col < dims.k && hi_col >= global_row {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + hi_col) * dims.m + global_row) as usize,
            )
        } else {
            0
        };
        let packed = lo as u32 | ((hi as u32) << 16);
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

fn stage_a_transposed_windowed_lower(
    src: &[u16],
    dst: &mut super::CtaATile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
    window: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, tile.row_base, k_base);
        let lo = if global_row < dims.m
            && global_col < dims.k
            && in_transposed_causal_window(global_row, global_col, window)
        {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + global_col) * dims.m + global_row) as usize,
            )
        } else {
            0
        };
        let hi_col = global_col + 1;
        let hi = if global_row < dims.m
            && hi_col < dims.k
            && in_transposed_causal_window(global_row, hi_col, window)
        {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + hi_col) * dims.m + global_row) as usize,
            )
        } else {
            0
        };
        let packed = lo as u32 | ((hi as u32) << 16);
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
fn in_causal_window(row: u32, col: u32, window: u32) -> bool {
    col <= row && row - col < window
}

#[inline(always)]
fn in_transposed_causal_window(row: u32, col: u32, window: u32) -> bool {
    col >= row && col - row < window
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
        let (global_row, global_col) = stage_coords(pair, tile.col_base, k_base);
        let lo = if global_row < dims.n && global_col < dims.k {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + global_col) * dims.n + global_row) as usize,
            )
        } else {
            0
        };
        let hi_col = global_col + 1;
        let hi = if global_row < dims.n && hi_col < dims.k {
            load_f16_global_bits_read_only(
                src.as_ptr(),
                ((tile.batch * dims.k + hi_col) * dims.n + global_row) as usize,
            )
        } else {
            0
        };
        let packed = lo as u32 | ((hi as u32) << 16);
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}
