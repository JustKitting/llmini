use cuda_device::SharedArray;

use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CtaTile};
use crate::kda_tc::swizzle_tc_offset;
use crate::mma::mma_m16n8k16_f16_f16_f32;

#[inline(always)]
pub(crate) fn mma_accumulate<const N_REPEATS: usize>(
    tile: CtaTile,
    a_tile: &SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    acc: &mut [[f32; 4]; N_REPEATS],
) {
    let a_fragments = load_a_fragments(a_tile, tile);
    let mut i = 0;
    while i < N_REPEATS {
        mma_m16n8k16_f16_f16_f32(
            a_fragments,
            load_b_fragments(b_tile, tile, tile.warp_n0 + i as u32),
            &mut acc[i],
        );
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn mma_accumulate_pair(
    tile: CtaTile,
    a_tile: &SharedArray<u16, CTA_A_ELEMS>,
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    acc0: &mut [f32; 4],
    acc1: &mut [f32; 4],
) {
    let a_fragments = load_a_fragments(a_tile, tile);
    mma_m16n8k16_f16_f16_f32(
        a_fragments,
        load_b_fragments(b_tile, tile, tile.warp_n0),
        acc0,
    );
    mma_m16n8k16_f16_f16_f32(
        a_fragments,
        load_b_fragments(b_tile, tile, tile.warp_n0 + 1),
        acc1,
    );
}

#[inline(always)]
fn load_a_fragments(a_tile: &SharedArray<u16, CTA_A_ELEMS>, tile: CtaTile) -> [u32; 4] {
    [
        load_a_fragment(a_tile, tile, 0),
        load_a_fragment(a_tile, tile, 1),
        load_a_fragment(a_tile, tile, 2),
        load_a_fragment(a_tile, tile, 3),
    ]
}

#[inline(always)]
fn load_b_fragments(
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    warp_n: u32,
) -> [u32; 2] {
    [
        load_b_fragment(b_tile, tile, warp_n, 0),
        load_b_fragment(b_tile, tile, warp_n, 1),
    ]
}

#[inline(always)]
fn load_a_fragment(a_tile: &SharedArray<u16, CTA_A_ELEMS>, tile: CtaTile, register: u32) -> u32 {
    let row = tile.warp_m * 16 + tile.group + if register & 1 == 0 { 0 } else { 8 };
    let col = tile.thread_in_group * 2 + if register < 2 { 0 } else { 8 };
    load_packed2(a_tile, row * 16 + col)
}

#[inline(always)]
fn load_b_fragment(
    b_tile: &SharedArray<u16, CTA_B_ELEMS>,
    tile: CtaTile,
    warp_n: u32,
    register: u32,
) -> u32 {
    let row = warp_n * 8 + tile.group;
    let col = tile.thread_in_group * 2 + if register == 0 { 0 } else { 8 };
    load_packed2(b_tile, row * 16 + col)
}

#[inline(always)]
fn load_packed2<const N: usize, const ALIGN: usize>(
    tile: &SharedArray<u16, N, ALIGN>,
    offset: u32,
) -> u32 {
    let offset = swizzle_tc_offset(offset);
    unsafe { *(tile.as_ptr().add(offset as usize) as *const u32) }
}
