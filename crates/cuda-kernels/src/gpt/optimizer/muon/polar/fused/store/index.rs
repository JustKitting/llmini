use crate::f16_tc_matmul::cta_tile::CtaTile;

#[inline(always)]
pub(super) fn row(tile: CtaTile, acc_index: usize) -> u32 {
    tile.row_base + tile.warp_m * 16 + tile.group + if acc_index < 2 { 0 } else { 8 }
}

#[inline(always)]
pub(super) fn col(tile: CtaTile, warp_n: u32, acc_index: usize) -> u32 {
    tile.col_base + warp_n * 8 + tile.thread_in_group * 2 + (acc_index as u32 & 1)
}
