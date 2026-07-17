use cuda_device::{DisjointSlice, thread};

use crate::kda_backward::{stage_compact_t_a, stage_hidden_dout_b_t, store_dh_quads};
use crate::kda_tc::{
    CompactStore::Add, CompactTileCtx, CtaATile, CtaBTile, KdaStateTile,
    stage_compact_a as stage_dm_compact_a, stage_compact_b_t_disjoint, stage_shared_state_b_t,
    store_compact_quads, tc_stage_loop,
};

use super::{KdaChunkwiseGrads, KdaChunkwiseInputs};

pub(super) fn add_kg_dh_to_du_tc(
    kg: &[f32],
    d_u: &mut DisjointSlice<f32>,
    d_h_next: &KdaStateTile,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
) {
    let mut acc = [[0.0_f32; 4]; 1];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < ctx.params.head_dim; {
        stage_dm_compact_a(kg, a_tile, ctx, k_base);
    } {
        stage_shared_state_b_t(d_h_next, b_tile, ctx, k_base);
    });

    store_compact_quads(acc, d_u, ctx, Add);
}

pub(super) fn compute_prev_dh_tc(
    inputs: KdaChunkwiseInputs<'_>,
    grads: &mut KdaChunkwiseGrads<'_>,
    states: (&KdaStateTile, &mut KdaStateTile),
    tiles: (&mut CtaATile, &mut CtaBTile),
    compact_ctx: CompactTileCtx<'_>,
) {
    let (d_h_next, d_h) = states;
    let (a_tile, b_tile) = tiles;
    let mut acc = [[0.0_f32; 4]; 1];

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < compact_ctx.params.chunk_size; {
        stage_compact_t_a(inputs.qg, a_tile, compact_ctx, k_base, 1.0);
    } {
        stage_hidden_dout_b_t(inputs.d_out, b_tile, compact_ctx, k_base);
    });

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < compact_ctx.params.chunk_size; {
        stage_compact_t_a(inputs.w, a_tile, compact_ctx, k_base, -1.0);
    } {
        stage_compact_b_t_disjoint(&mut grads.u_to_du, b_tile, compact_ctx, k_base);
    });

    store_dh_quads(acc, d_h_next, d_h, inputs.g, compact_ctx);
}
