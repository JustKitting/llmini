use cuda_device::DisjointSlice;

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
    store_one::<STRICT, NEGATE>(acc[0], tile, warp_n, 0, out, rows, cols);
    store_one::<STRICT, NEGATE>(acc[1], tile, warp_n, 1, out, rows, cols);
    store_one::<STRICT, NEGATE>(acc[2], tile, warp_n, 2, out, rows, cols);
    store_one::<STRICT, NEGATE>(acc[3], tile, warp_n, 3, out, rows, cols);
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
