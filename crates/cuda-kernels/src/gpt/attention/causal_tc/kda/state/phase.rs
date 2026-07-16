use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, thread};

use super::super::super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::f16_tc_matmul::convert::store_f16x2_shared;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_K};
use crate::kda_common::{chunk_g_last_index, compact_index, kda_decay_exp, state_elems};
use crate::kda_tc::KdaDecayTile;
use crate::kda_tc::{
    CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, add_shared_state_quads, stage_compact_a,
    stage_compact_b_t_disjoint as stage_vnew_b_t_disjoint, stage_shared_state_b_t,
    store_vnew_quads, tc_stage_loop,
};

pub(super) fn compute_ws_to_vnew(
    w: &[f32],
    u: &[f32],
    v_new: &mut DisjointSlice<f32>,
    state: &KdaStateTile,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 2];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.head_dim; {
        stage_compact_a(w, a_tile, ctx, k_base);
    } {
        stage_shared_state_b_t(state, b_tile, ctx, k_base);
    });
    store_vnew_quads(acc, u, v_new, ctx);
}

pub(super) fn compute_kg_vnew_add_state(
    k: &[f32],
    v_new: &mut DisjointSlice<f32>,
    state: &mut KdaStateTile,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 2];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.chunk_size; {
        stage_kg_t_a(k, a_tile, ctx, k_base);
    } {
        stage_vnew_b_t_disjoint(v_new, b_tile, ctx, k_base);
    });
    add_shared_state_quads(acc, ctx.tile, state, ctx.params);
}

pub(super) fn decay_state(
    state: &mut KdaStateTile,
    decay: &mut KdaDecayTile,
    chunk_g_last: &[f32],
    bh: u32,
    chunk: u32,
    tid: u32,
    ctx: CompactTileCtx<'_>,
) {
    let state_elems = state_elems(ctx.params);
    if tid < ctx.params.head_dim {
        let g_last = chunk_g_last[chunk_g_last_index(bh, chunk, tid, ctx.params)];
        decay[tid as usize] = kda_decay_exp(g_last);
    }
    thread::sync_threads();

    let mut linear = tid;
    while linear < state_elems {
        let k_dim = linear / ctx.params.head_dim;
        state[linear as usize] *= decay[k_dim as usize];
        linear += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn stage_kg_t_a(src: &[f32], a_tile: &mut CtaATile, ctx: CompactTileCtx<'_>, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let dim = ctx.tile.row_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if dim < ctx.params.head_dim && token0 < ctx.end {
            src[compact_index(ctx.batch, token0, ctx.head, dim, ctx.params)]
        } else {
            0.0
        };
        let hi = if dim < ctx.params.head_dim && token1 < ctx.end {
            src[compact_index(ctx.batch, token1, ctx.head, dim, ctx.params)]
        } else {
            0.0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += thread::blockDim_x() * 2;
    }
}
