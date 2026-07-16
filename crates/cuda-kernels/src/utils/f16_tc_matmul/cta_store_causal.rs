use cuda_device::DisjointSlice;

use super::convert::store_f32x2_global;
use super::cta_tile::CtaTile;

#[inline(always)]
pub(super) fn store_lower(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_tile::<false, false>(acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
pub(super) fn store_strict_lower(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_tile::<true, false>(acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
pub(super) fn store_strict_neg(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_tile::<true, true>(acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
fn store_tile<const STRICT: bool, const NEGATE: bool>(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_pair::<STRICT, NEGATE>(acc[0], acc[1], tile, warp_n, 0, out, rows, cols);
    store_pair::<STRICT, NEGATE>(acc[2], acc[3], tile, warp_n, 2, out, rows, cols);
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA store coordinates are explicit"
)]
fn store_pair<const STRICT: bool, const NEGATE: bool>(
    lo: f32,
    hi: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    let (row0, col0) = tile.accumulator_coords(warp_n, acc_index);
    let (row1, col1) = tile.accumulator_coords(warp_n, acc_index + 1);
    if row0 < rows && row1 == row0 && col0 + 1 == col1 && col1 < cols {
        let lower0 = if STRICT { row0 > col0 } else { row0 >= col0 };
        let lower1 = if STRICT { row1 > col1 } else { row1 >= col1 };
        let value0 = if lower0 {
            if NEGATE { -lo } else { lo }
        } else {
            0.0
        };
        let value1 = if lower1 {
            if NEGATE { -hi } else { hi }
        } else {
            0.0
        };
        let index = ((tile.batch * rows + row0) * cols + col0) as usize;
        store_f32x2_global(out.as_mut_ptr(), index, value0, value1);
    } else {
        store_one::<STRICT, NEGATE>(lo, tile, warp_n, acc_index, out, rows, cols);
        store_one::<STRICT, NEGATE>(hi, tile, warp_n, acc_index + 1, out, rows, cols);
    }
}

#[inline(always)]
fn store_one<const STRICT: bool, const NEGATE: bool>(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    let (row, col) = tile.accumulator_coords(warp_n, acc_index);
    if row < rows && col < cols {
        let lower = if STRICT { row > col } else { row >= col };
        let value = if lower {
            if NEGATE { -acc } else { acc }
        } else {
            0.0
        };
        unsafe {
            *out.get_unchecked_mut(((tile.batch * rows + row) * cols + col) as usize) = value;
        }
    }
}
