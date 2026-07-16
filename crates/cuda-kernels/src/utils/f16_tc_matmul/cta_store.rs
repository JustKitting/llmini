use cuda_device::DisjointSlice;

use super::convert::store_f32x2_global;
use super::cta_tile::CtaTile;

#[inline(always)]
pub(super) fn store(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_tile::<true>(acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
pub(super) fn store_aligned(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_tile::<false>(acc, tile, warp_n, out, rows, cols);
}

#[inline(always)]
fn store_tile<const CHECK_BOUNDS: bool>(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    store_pair::<CHECK_BOUNDS>(acc[0], acc[1], tile, warp_n, 0, out, rows, cols);
    store_pair::<CHECK_BOUNDS>(acc[2], acc[3], tile, warp_n, 2, out, rows, cols);
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA store coordinates are explicit"
)]
fn store_pair<const CHECK_BOUNDS: bool>(
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
    if !CHECK_BOUNDS || (row0 < rows && row1 == row0 && col0 + 1 == col1 && col1 < cols) {
        let index = ((tile.batch * rows + row0) * cols + col0) as usize;
        store_f32x2_global(out.as_mut_ptr(), index, lo, hi);
    } else {
        store_one::<CHECK_BOUNDS>(lo, tile, warp_n, acc_index, out, rows, cols);
        store_one::<CHECK_BOUNDS>(hi, tile, warp_n, acc_index + 1, out, rows, cols);
    }
}

#[inline(always)]
fn store_one<const CHECK_BOUNDS: bool>(
    acc: f32,
    tile: CtaTile,
    warp_n: u32,
    acc_index: usize,
    out: &mut DisjointSlice<f32>,
    rows: u32,
    cols: u32,
) {
    let (row, col) = tile.accumulator_coords(warp_n, acc_index);
    if !CHECK_BOUNDS || (row < rows && col < cols) {
        unsafe {
            *out.get_unchecked_mut(((tile.batch * rows + row) * cols + col) as usize) = acc;
        }
    }
}
