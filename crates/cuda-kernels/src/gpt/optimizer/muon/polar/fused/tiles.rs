use cuda_device::{SharedArray, thread};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_M, CTA_N, CtaTile};

use super::super::super::super::work_grid::WorkGrid;
use super::store::store_plain_tile;

mod compute;
use compute::compute_tile;
mod symmetric;
pub(super) use symmetric::{run_symmetric_polynomial_tiles, run_symmetric_tiles};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn run_plain_tiles(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    work: WorkGrid,
    m: u32,
    n: u32,
    k: u32,
    rhs_transposed: bool,
) {
    let n_tiles = n.div_ceil(CTA_N);
    let tile_count = m.div_ceil(CTA_M) * n_tiles;
    let mut tile_index = work.block();

    while tile_index < tile_count {
        let tile_row = tile_index / n_tiles;
        let tile_col = tile_index - tile_row * n_tiles;
        let tile = CtaTile::from_tile(thread::threadIdx_x(), tile_row, tile_col, 0);
        let acc = compute_tile(a, b, a_tile, b_tile, tile, m, n, k, rhs_transposed);
        store_plain_tile(acc, tile, out, m, n);
        tile_index += work.blocks();
    }
}
