mod phase;

use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::store_f32x2_global;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_backward::load_chunk_state;
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape, state_elems};
use crate::kda_tc::{CompactTileCtx, CtaATile, CtaBTile, KdaStateTile};

use phase::{add_kg_dh_to_du_tc, compute_prev_dh_tc};

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkwiseInputs<'a> {
    pub(crate) qg: &'a [f32],
    pub(crate) kg: &'a [f32],
    pub(crate) w: &'a [f32],
    pub(crate) g: &'a [f32],
    pub(crate) chunk_states: &'a [u16],
    pub(crate) d_out: &'a [f32],
}
pub(crate) struct KdaChunkwiseGrads<'a> {
    pub(crate) u_to_du: DisjointSlice<'a, f32>,
    pub(crate) d_h_states: DisjointSlice<'a, f32>,
}
pub(crate) fn chunkwise_kda_backward_body(
    inputs: KdaChunkwiseInputs<'_>,
    mut grads: KdaChunkwiseGrads<'_>,
    params: CausalAttentionParams,
    states: (&mut KdaStateTile, &mut KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaBTile),
) {
    let (state, mut d_h_next, mut d_h) = states;
    let (a_tile, b_tile) = tiles;
    let bh = thread::blockIdx_x();
    let tid = thread::threadIdx_x();
    if bh >= batch_head(&params) || !kda_tc_shape(&params) {
        return;
    }

    let state_elems = state_elems(&params);
    let chunks = chunk_count(&params);
    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;

    let mut idx = tid;
    while idx < state_elems {
        d_h_next[idx as usize] = 0.0;
        idx += thread::blockDim_x();
    }
    thread::sync_threads();

    let mut chunk_remaining = chunks;
    while chunk_remaining > 0 {
        let chunk = chunk_remaining - 1;
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        load_chunk_state(
            inputs.chunk_states,
            state,
            bh,
            chunk,
            state_elems,
            &params,
            thread::blockDim_x(),
        );

        let mut pair = tid * 2;
        let d_h_base = ((bh * chunks + chunk) * state_elems) as usize;
        let d_h_states_ptr = grads.d_h_states.as_mut_ptr();
        while pair < state_elems {
            store_f32x2_global(
                d_h_states_ptr,
                d_h_base + pair as usize,
                d_h_next[pair as usize],
                d_h_next[pair as usize + 1],
            );
            pair += thread::blockDim_x() * 2;
        }

        let tile = CtaTile::from_ultra_wide_tile(tid, 0, 0, 0);
        let ctx = CompactTileCtx::new(tile, (batch, head), (start, end), &params);
        add_kg_dh_to_du_tc(inputs.kg, &mut grads.u_to_du, d_h_next, a_tile, b_tile, ctx);
        thread::sync_threads();

        compute_prev_dh_tc(
            inputs,
            &mut grads,
            (&*d_h_next, &mut *d_h),
            (&mut *a_tile, &mut *b_tile),
            ctx,
        );
        thread::sync_threads();

        let completed = d_h_next;
        d_h_next = d_h;
        d_h = completed;

        chunk_remaining -= 1;
    }
}
