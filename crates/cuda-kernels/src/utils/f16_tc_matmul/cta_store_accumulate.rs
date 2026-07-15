use cuda_device::DisjointSlice;

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
    store_one(acc[0], tile, warp_n, 0, out, rows, cols);
    store_one(acc[1], tile, warp_n, 1, out, rows, cols);
    store_one(acc[2], tile, warp_n, 2, out, rows, cols);
    store_one(acc[3], tile, warp_n, 3, out, rows, cols);
}

#[inline(always)]
fn store_one(
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
        let offset = ((tile.batch * rows + row) * cols + col) as usize;
        unsafe {
            *out.get_unchecked_mut(offset) += acc;
        }
    }
}
