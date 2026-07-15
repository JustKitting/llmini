use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::float_ptx::fma_f32;

use super::super::coefficients::Coefficients;
use super::index::{col, row};

macro_rules! store_acc4 {
    ($store:ident, $acc:ident, $tile:expr, $warp_n:expr, $($arg:expr),+ $(,)?) => {{
        $store($acc[0], $tile, $warp_n, 0, $($arg),+);
        $store($acc[1], $tile, $warp_n, 1, $($arg),+);
        $store($acc[2], $tile, $warp_n, 2, $($arg),+);
        $store($acc[3], $tile, $warp_n, 3, $($arg),+);
    }};
}

macro_rules! store_tile_acc4 {
    ($store:ident, $acc:ident, $tile:expr, $($arg:expr),+ $(,)?) => {{
        $store($acc[0], $tile, $tile.warp_n0, $($arg),+); $store($acc[1], $tile, $tile.warp_n0 + 1, $($arg),+); $store($acc[2], $tile, $tile.warp_n0 + 2, $($arg),+); $store($acc[3], $tile, $tile.warp_n0 + 3, $($arg),+);
    }};
}

#[inline(always)]
pub(crate) fn store_plain(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: *mut f32,
    rows: u32,
    cols: u32,
) {
    store_acc4!(store_plain_one, acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
pub(crate) fn store_plain_tile(
    acc: [[f32; 4]; 4],
    tile: CtaTile,
    out: *mut f32,
    rows: u32,
    cols: u32,
) {
    store_tile_acc4!(store_plain, acc, tile, out, rows, cols);
}

#[inline(always)]
pub(crate) fn store_plain_transposed(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: *mut f32,
    dim: u32,
) {
    store_acc4!(store_plain_transposed_one, acc, tile, warp_n, out, dim);
}

#[inline(always)]
pub(crate) fn store_plain_transposed_tile(
    acc: [[f32; 4]; 4],
    tile: CtaTile,
    out: *mut f32,
    dim: u32,
) {
    store_tile_acc4!(store_plain_transposed, acc, tile, out, dim);
}

#[inline(always)]
pub(crate) fn store_symmetric_polynomial(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    base: *const f32,
    out: *mut f32,
    dim: u32,
    coefficients: Coefficients,
) {
    store_acc4!(
        store_symmetric_polynomial_one,
        acc,
        tile,
        warp_n,
        base,
        out,
        dim,
        coefficients
    );
}

#[inline(always)]
pub(crate) fn store_symmetric_polynomial_tile(
    acc: [[f32; 4]; 4],
    tile: CtaTile,
    base: *const f32,
    out: *mut f32,
    dim: u32,
    coefficients: Coefficients,
) {
    store_tile_acc4!(
        store_symmetric_polynomial,
        acc,
        tile,
        base,
        out,
        dim,
        coefficients
    );
}

#[inline(always)]
fn store_plain_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: *mut f32,
    rows: u32,
    cols: u32,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < rows && col < cols {
        write_f32(out, row * cols + col, acc);
    }
}

#[inline(always)]
fn store_plain_transposed_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: *mut f32,
    dim: u32,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < dim && col < dim {
        write_f32(out, col * dim + row, acc);
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
fn store_symmetric_polynomial_one(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    base: *const f32,
    out: *mut f32,
    dim: u32,
    coefficients: Coefficients,
) {
    let row = row(tile, acc_index);
    let col = col(tile, warp_n, acc_index);
    if row < dim && col < dim {
        let offset = row * dim + col;
        write_f32(
            out,
            offset,
            polynomial_value(acc, read_f32(base, offset), row == col, coefficients),
        );
    }
    if row < dim && col < dim && row != col {
        let offset = col * dim + row;
        write_f32(
            out,
            offset,
            polynomial_value(acc, read_f32(base, offset), false, coefficients),
        );
    }
}

#[inline(always)]
fn polynomial_value(acc: f32, base: f32, diagonal: bool, coefficients: Coefficients) -> f32 {
    let identity = if diagonal { coefficients.a } else { 0.0 };
    fma_f32(coefficients.c, acc, fma_f32(coefficients.b, base, identity))
}
