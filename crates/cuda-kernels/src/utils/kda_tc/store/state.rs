use cuda_device::SharedArray;

use super::coords::compact_fragment_coords;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::cta_tile::CtaTile;

pub(crate) fn add_shared_state_quads<const STATE_ELEMS: usize, const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    tile: CtaTile,
    state: &mut SharedArray<f32, STATE_ELEMS>,
    params: &CausalAttentionParams,
) {
    for_acc_fragments!(acc, tile, |warp_n, frag, value| {
        let (k_dim, v_dim) = compact_fragment_coords(tile, warp_n, frag);
        if k_dim < params.head_dim && v_dim < params.head_dim {
            state[(k_dim * params.head_dim + v_dim) as usize] += value;
        }
    });
}
