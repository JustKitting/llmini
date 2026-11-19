mod phase;

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::store_f16x2_global;
use crate::f16_tc_matmul::cta_tile::CtaTile;
use crate::kda_common::{batch_head, chunk_count, kda_tc_shape, state_elems};
use crate::kda_tc::{CompactTileCtx, CtaATile, CtaBTile, KdaStateTile};
use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, thread};
use phase::{compute_prev_dh_tc, finish_du, project_du_and_state_seed};

#[derive(Clone, Copy)]
pub(crate) struct KdaChunkwiseInputs<'a> {
    pub(crate) qg: &'a [f32],
    pub(crate) kg: &'a [f32],
    pub(crate) w: &'a [u16],
    pub(crate) g: &'a [f32],
    pub(crate) aqk: &'a [u16],
    pub(crate) d_out: &'a [u16],
}
pub(crate) struct KdaChunkwiseGrads<'a> {
    pub(crate) d_u: DisjointSlice<'a, u16>,
    pub(crate) d_h_states: DisjointSlice<'a, u16>,
}
pub(crate) fn chunkwise_kda_backward_body(
    inputs: KdaChunkwiseInputs<'_>,
    mut grads: KdaChunkwiseGrads<'_>,
    params: CausalAttentionParams,
    states: (&mut KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaATile, &mut CtaBTile),
) {
    let (mut d_h_next, mut d_h) = states;
    let (a_tile, q_tile, b_tile) = tiles;
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

    let mut remaining = chunks;
    while remaining > 0 {
        let chunk = remaining - 1;
        let start = chunk * params.chunk_size;
        let end = params.seq_len.min(start + params.chunk_size);
        // The previous implementation loaded a forward state here, but no
        // computation consumed that copy. Only the two FP32 derivative carries
        // below are live in this recurrence.
        let mut pair = tid * 2;
        let base = ((bh * chunks + chunk) * state_elems) as usize;
        while pair < state_elems {
            store_f16x2_global(
                grads.d_h_states.as_mut_ptr(),
                base + pair as usize,
                cvt_f16x2_f32(d_h_next[pair as usize], d_h_next[pair as usize + 1]),
            );
            pair += thread::blockDim_x() * 2;
        }
        let tile = CtaTile::from_ultra_wide_tile(tid, 0, 0, 0);
        let ctx = CompactTileCtx::new(tile, (batch, head), (start, end), &params);
        let (du_initial, state_seed) =
            project_du_and_state_seed(inputs, (a_tile, q_tile, b_tile), ctx);
        finish_du(
            inputs.kg,
            &mut grads.d_u,
            d_h_next,
            du_initial,
            (a_tile, b_tile),
            ctx,
        );
        thread::sync_threads();

        compute_prev_dh_tc(
            inputs,
            &mut grads,
            state_seed,
            (&*d_h_next, &mut *d_h),
            (a_tile, b_tile),
            ctx,
        );
        thread::sync_threads();
        let completed = d_h_next;
        d_h_next = d_h;
        d_h = completed;

        remaining -= 1;
    }
}
