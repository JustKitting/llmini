use cuda_device::{SharedArray, grid};

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;

mod coefficients;
mod normalize;
mod ptr;
mod stage;
mod store;
mod tiles;

pub use coefficients::{Coefficients, coefficients};
pub(crate) use normalize::normalize_source_to_x;
use ptr::{source_ptr, target_ptr};
use tiles::{run_plain_tiles, run_symmetric_polynomial_tiles, run_symmetric_tiles};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(crate) fn polar_express_from_source_ptr(
    source: *const f32,
    x_ptr: *mut f32,
    next_ptr: *mut f32,
    gram_ptr: *mut f32,
    ax_ptr: *mut f32,
    chunk_ptr: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
    source_rows: u32,
    source_cols: u32,
    transpose_source: bool,
    iterations: u32,
) -> *const f32 {
    let polar_rows = if transpose_source {
        source_cols
    } else {
        source_rows
    };
    let polar_cols = if transpose_source {
        source_rows
    } else {
        source_cols
    };

    normalize_source_to_x(
        source,
        x_ptr,
        chunk_ptr,
        warp_sums,
        work,
        source_rows,
        source_cols,
        polar_rows,
        polar_cols,
        transpose_source,
    );
    grid::sync();

    let mut iter = 0;
    while iter < iterations {
        let source = source_ptr(iter, x_ptr, next_ptr);
        let target = target_ptr(iter, x_ptr, next_ptr);

        run_symmetric_tiles(
            source, gram_ptr, a_tile, b_tile, work, polar_rows, polar_cols,
        );
        grid::sync();

        run_symmetric_polynomial_tiles(
            gram_ptr,
            gram_ptr,
            ax_ptr,
            a_tile,
            b_tile,
            work,
            polar_rows,
            coefficients(iter),
        );
        grid::sync();

        run_plain_tiles(
            ax_ptr, source, target, a_tile, b_tile, work, polar_rows, polar_cols, polar_rows, true,
        );
        grid::sync();

        iter += 1;
    }

    source_ptr(iterations, x_ptr, next_ptr)
}
