use cuda_device::{SharedArray, thread};

use crate::device_ptr::read_f32;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaTile};

macro_rules! stage_four_offsets {
    ($stage:ident($($arg:expr),+), $offset:expr) => {{
        $stage($($arg,)+ $offset);
        $stage($($arg,)+ $offset + CTA_THREADS);
        $stage($($arg,)+ $offset + CTA_THREADS * 2);
        $stage($($arg,)+ $offset + CTA_THREADS * 3);
    }};
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn stage_tiles(
    a: *const f32,
    b: *const f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    m: u32,
    n: u32,
    k: u32,
    k_base: u32,
    rhs_transposed: bool,
) {
    let offset = thread::threadIdx_x();
    stage_four_offsets!(stage_a(a, a_tile, tile, m, k, k_base), offset);
    stage_four_offsets!(
        stage_b(b, b_tile, tile, n, k, k_base, rhs_transposed),
        offset
    );
}

#[inline(always)]
fn stage_a(
    a: *const f32,
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    tile: CtaTile,
    m: u32,
    k: u32,
    k_base: u32,
    offset: u32,
) {
    let row = offset / CTA_K;
    let col = offset - row * CTA_K;
    let global_row = tile.row_base + row;
    let global_col = k_base + col;
    a_tile[offset as usize] = if global_row < m && global_col < k {
        cvt_rn_f16_f32(read_f32(a, global_row * k + global_col))
    } else {
        0
    };
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
fn stage_b(
    b: *const f32,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    n: u32,
    k: u32,
    k_base: u32,
    rhs_transposed: bool,
    offset: u32,
) {
    let row = offset / CTA_K;
    let col = offset - row * CTA_K;
    let global_row = tile.col_base + row;
    let global_col = k_base + col;
    b_tile[offset as usize] = if global_row < n && global_col < k {
        let index = if rhs_transposed {
            global_col * n + global_row
        } else {
            global_row * k + global_col
        };
        cvt_rn_f16_f32(read_f32(b, index))
    } else {
        0
    };
}
