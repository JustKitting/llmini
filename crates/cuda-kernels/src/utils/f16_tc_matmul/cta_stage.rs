use cuda_device::{SharedArray, thread};

use super::convert::{load_f16x2_global_bits, store_f16x2_shared};
use super::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS, CtaMatmulDims, CtaTile};

macro_rules! stage_tiles_fn {
    ($name:ident, $check_bounds:expr) => {
        pub(super) fn $name(
            a: &[u16],
            b_t: &[u16],
            a_tile: &mut super::CtaATile,
            b_tile: &mut super::CtaBTile,
            tile: CtaTile,
            dims: CtaMatmulDims,
            k_base: u32,
        ) {
            stage_tiles_impl::<$check_bounds>(a, b_t, a_tile, b_tile, tile, dims, k_base);
        }
    };
}

stage_tiles_fn!(stage_tiles, true);
stage_tiles_fn!(stage_tiles_aligned, false);

fn stage_tiles_impl<const CHECK_BOUNDS: bool>(
    a: &[u16],
    b_t: &[u16],
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    tile: CtaTile,
    dims: CtaMatmulDims,
    k_base: u32,
) {
    stage_matrix_tile::<CHECK_BOUNDS, CTA_A_ELEMS>(
        a,
        a_tile,
        tile,
        tile.row_base,
        dims.m,
        dims.k,
        k_base,
    );
    stage_matrix_tile::<CHECK_BOUNDS, CTA_B_ELEMS>(
        b_t,
        b_tile,
        tile,
        tile.col_base,
        dims.n,
        dims.k,
        k_base,
    );
}

fn stage_matrix_tile<const CHECK_BOUNDS: bool, const TILE_ELEMS: usize>(
    src: &[u16],
    dst: &mut SharedArray<u16, TILE_ELEMS>,
    tile: CtaTile,
    row_base: u32,
    rows: u32,
    cols: u32,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < TILE_ELEMS as u32 {
        let (global_row, global_col) = stage_coords(pair, row_base, k_base);
        let packed = if !CHECK_BOUNDS || (global_row < rows && global_col + 1 < cols) {
            let index = ((tile.batch * rows + global_row) * cols + global_col) as usize;
            load_f16x2_global_bits(src.as_ptr(), index)
        } else if global_row < rows && global_col < cols {
            src[((tile.batch * rows + global_row) * cols + global_col) as usize] as u32
        } else {
            0
        };
        store_f16x2_shared(dst.as_mut_ptr(), pair as usize, packed);
        pair += CTA_THREADS * 2;
    }
}

#[inline(always)]
pub(crate) fn stage_coords(offset: u32, row_base: u32, k_base: u32) -> (u32, u32) {
    let row = offset / CTA_K;
    let col = offset - row * CTA_K;
    (row_base + row, k_base + col)
}

#[inline(always)]
pub(crate) fn load_a_fragments(a_tile: &super::CtaATile, tile: CtaTile) -> [u32; 4] {
    [
        load_a_fragment(a_tile, tile, 0),
        load_a_fragment(a_tile, tile, 1),
        load_a_fragment(a_tile, tile, 2),
        load_a_fragment(a_tile, tile, 3),
    ]
}

#[inline(always)]
pub(crate) fn load_b_fragments(b_tile: &super::CtaBTile, tile: CtaTile, warp_n: u32) -> [u32; 2] {
    [
        load_b_fragment(b_tile, tile, warp_n, 0),
        load_b_fragment(b_tile, tile, warp_n, 1),
    ]
}

#[inline(always)]
fn load_a_fragment(a_tile: &super::CtaATile, tile: CtaTile, register: u32) -> u32 {
    let row = tile.warp_m * 16 + tile.group + if register & 1 == 0 { 0 } else { 8 };
    let col = tile.thread_in_group * 2 + if register < 2 { 0 } else { 8 };
    load_packed2(a_tile, row * CTA_K + col)
}

#[inline(always)]
fn load_b_fragment(b_tile: &super::CtaBTile, tile: CtaTile, warp_n: u32, register: u32) -> u32 {
    let row = warp_n * 8 + tile.group;
    let col = tile.thread_in_group * 2 + if register == 0 { 0 } else { 8 };
    load_packed2(b_tile, row * CTA_K + col)
}

#[inline(always)]
fn load_packed2<const N: usize>(tile: &SharedArray<u16, N>, offset: u32) -> u32 {
    (tile[offset as usize] as u32) | ((tile[offset as usize + 1] as u32) << 16)
}
