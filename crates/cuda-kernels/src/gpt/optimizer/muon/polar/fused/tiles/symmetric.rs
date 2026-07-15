#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_M, CtaTile};
use crate::float_ptx::sqrt_f32;

use super::super::super::super::super::work_grid::WorkGrid;
use super::super::coefficients::Coefficients;
use super::super::store::{
    store_plain_tile, store_plain_transposed_tile, store_symmetric_polynomial_tile,
};
use super::compute_tile;

macro_rules! for_upper_triangle_tiles {
    ($work:expr, $dim:expr, |$tile_row:ident, $tile_col:ident| $body:block) => {{
        let tile_dim = $dim.div_ceil(CTA_M);
        let tile_count = tile_dim * (tile_dim + 1) / 2;
        let mut tile_index = $work.block();
        while tile_index < tile_count {
            let ($tile_row, $tile_col) = upper_triangle_tile(tile_index, tile_dim);
            $body
            tile_index += $work.blocks();
        }
    }};
}

pub(crate) fn run_symmetric_tiles(
    source: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    dim: u32,
    k: u32,
) {
    for_upper_triangle_tiles!(work, dim, |tile_row, tile_col| {
        run_tile(source, out, a_tile, b_tile, dim, k, tile_row, tile_col);
    });
}

pub(crate) fn run_symmetric_polynomial_tiles(
    source: *const f32,
    base: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    dim: u32,
    coefficients: Coefficients,
) {
    for_upper_triangle_tiles!(work, dim, |tile_row, tile_col| {
        run_polynomial_tile(
            source,
            base,
            out,
            a_tile,
            b_tile,
            dim,
            tile_row,
            tile_col,
            coefficients,
        );
    });
}

#[inline(always)]
fn upper_triangle_tile(index: u32, tile_dim: u32) -> (u32, u32) {
    let n = (2 * tile_dim + 1) as f32;
    let row = ((n - sqrt_f32(n * n - 8.0 * index as f32)) * 0.5) as u32;
    let row_start = row * (2 * tile_dim - row + 1) / 2;
    let col = row + (index - row_start);
    (row, col)
}

fn run_tile(
    source: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    dim: u32,
    k: u32,
    tile_row: u32,
    tile_col: u32,
) {
    let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
    let acc = compute_tile(source, source, a_tile, b_tile, tile, dim, dim, k, false);
    store_plain_tile(acc, tile, out, dim, dim);
    if tile_col != tile_row {
        store_plain_transposed_tile(acc, tile, out, dim);
    }
}

fn run_polynomial_tile(
    source: *const f32,
    base: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    dim: u32,
    tile_row: u32,
    tile_col: u32,
    coefficients: Coefficients,
) {
    let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
    let acc = compute_tile(source, source, a_tile, b_tile, tile, dim, dim, dim, false);
    store_symmetric_polynomial_tile(acc, tile, base, out, dim, coefficients);
}
