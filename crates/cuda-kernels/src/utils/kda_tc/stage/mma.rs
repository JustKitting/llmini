use cuda_device::SharedArray;

use crate::f16_tc_matmul::cta_stage::{load_a_fragments, load_b_fragments};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CtaTile};
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
