mod phase;

use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape, state_elems};
use crate::kda_tc::{CompactTileCtx, CtaTiles, KdaDecayTile, KdaStateTile};

use phase::{compute_kg_vnew_add_state, compute_ws_to_vnew, decay_state};

#[derive(Clone, Copy)]
pub(in super::super) struct KdaStateSaveInputs<'a> {
    pub(in super::super) kg: &'a [f32],
    pub(in super::super) w: &'a [f32],
    pub(in super::super) u: &'a [f32],
    pub(in super::super) chunk_g_last: &'a [f32],
}

pub(in super::super) fn chunk_kda_state_save_body(
    inputs: KdaStateSaveInputs<'_>,
    mut v_new: DisjointSlice<f32>,
    mut chunk_states: DisjointSlice<u16>,
    params: CausalAttentionParams,
    state: &mut KdaStateTile,
    decay: &mut KdaDecayTile,
    tiles: CtaTiles<'_>,
) {
    let (a_tile, b_tile) = tiles;
    let bh = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if bh >= batch_head(&params) || !kda_tc_shape(&params) {
        return;
    }
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;

    let state_elems = state_elems(&params);
    let chunks = chunk_count(&params);
    if chunks == 0 {
        return;
    }
    let first_state_base = (bh * chunks * state_elems) as usize;
    let mut linear = tid;
    while linear < state_elems {
        state[linear as usize] = 0.0;
        unsafe {
            *chunk_states.get_unchecked_mut(first_state_base + linear as usize) = 0;
        }
        linear += thread::blockDim_x();
    }
    thread::sync_threads();

    let tile = CtaTile::from_ultra_wide_tile(tid, 0, 0, 0);
    let mut chunk = 0;
    while chunk < chunks {
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        let ctx = CompactTileCtx::new(tile, (batch, head), (start, end), &params);

        if chunk != 0 {
            linear = tid;
            while linear < state_elems {
                let base = ((bh * chunks + chunk) * state_elems) as usize;
                unsafe {
                    *chunk_states.get_unchecked_mut(base + linear as usize) =
                        cvt_rn_f16_f32(state[linear as usize]);
                }
                linear += thread::blockDim_x();
            }
            thread::sync_threads();
        }

        compute_ws_to_vnew(inputs.w, inputs.u, &mut v_new, state, a_tile, b_tile, ctx);
        thread::sync_threads();

        decay_state(state, decay, inputs.chunk_g_last, bh, chunk, tid, ctx);
        thread::sync_threads();
        compute_kg_vnew_add_state(inputs.kg, &mut v_new, state, a_tile, b_tile, ctx);
        thread::sync_threads();

        chunk += 1;
    }
}
