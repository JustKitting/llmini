use cuda_device::SharedArray;

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS};

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;
use super::super::polar::fused::polar_express_from_source_ptr;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn run_polar_step(
    oriented_ptr: *mut f32,
    scaled_ptr: *mut f32,
    polar_x: *mut f32,
    polar_gram: *mut f32,
    polar_ax: *mut f32,
    polar_chunks: *mut f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
    rows: u32,
    cols: u32,
    transposed: bool,
    iterations: u32,
) -> *const f32 {
    let oriented_rows = if transposed { cols } else { rows };
    let oriented_cols = if transposed { rows } else { cols };
    let needs_transpose = oriented_rows > oriented_cols;
    // Square matrices do not transpose during normalization, so the initial
    // Polar buffer can reuse oriented in-place. Rectangular matrices still need
    // a separate buffer because normalization transposes the source.
    let polar_x = if needs_transpose {
        polar_x
    } else {
        oriented_ptr
    };
    let polar_next = if needs_transpose {
        oriented_ptr
    } else {
        scaled_ptr
    };
    let polar_ax = if needs_transpose {
        scaled_ptr
    } else {
        polar_ax
    };

    polar_express_from_source_ptr(
        oriented_ptr,
        polar_x,
        polar_next,
        polar_gram,
        polar_ax,
        polar_chunks,
        a_tile,
        b_tile,
        warp_sums,
        work,
        oriented_rows,
        oriented_cols,
        needs_transpose,
        iterations,
    )
}
