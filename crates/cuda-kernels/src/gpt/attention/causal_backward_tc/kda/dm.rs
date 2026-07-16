use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{load_f32x2_global, store_f16x2_shared};
use crate::f16_tc_matmul::cta_tile::{CTA_B_ELEMS, CTA_K};
use crate::kda_common::{beta_compact_index, compact_index, kda_decay_exp};
use crate::kda_tc::{
    CompactTileCtx, CtaBTile, CtaTiles, KdaChunkTileCtx, stage_compact_a as stage_dm_compact_a,
    stage_compact_token_dim_b_t as stage_dm_compact_b_t, store_chunk_matrix_quads, tc_stage_loop,
};

#[derive(Clone, Copy)]
pub(crate) struct KdaDmInputs<'a> {
    pub(crate) kg: &'a [f32],
    pub(crate) vbeta: &'a [f32],
    pub(crate) g: &'a [f32],
    pub(crate) beta: &'a [f32],
    pub(crate) d_u: &'a [f32],
    pub(crate) d_w: &'a [f32],
}

pub(crate) fn chunk_intra_kda_dm_body(
    inputs: KdaDmInputs<'_>,
    mut d_m: DisjointSlice<f32>,
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    let Some(ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    let (a_tile, b_tile) = tiles;
    let compact_ctx = ctx.compact;
    let mut acc = [[0.0_f32; 4]; 2];

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(inputs.d_w, a_tile, compact_ctx, k_base);
    } {
        stage_dm_kpos_b_t(inputs, b_tile, compact_ctx, k_base);
    });

    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(inputs.d_u, a_tile, compact_ctx, k_base);
    } {
        stage_dm_compact_b_t(inputs.vbeta, b_tile, compact_ctx, k_base);
    });

    store_chunk_matrix_quads(acc, &mut d_m, ctx.matrix);
}

fn stage_dm_kpos_b_t(
    inputs: KdaDmInputs<'_>,
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let source = ctx.tile.col_base + row;
        let dim = k_base + col;
        let token = ctx.start + source;
        let packed = if token < ctx.end && dim + 1 < ctx.params.head_dim {
            let compact = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            let last = compact_index(ctx.batch, ctx.end - 1, ctx.head, dim, ctx.params);
            let (g0, g1) = load_f32x2_global(inputs.g.as_ptr(), compact);
            let (kg0, kg1) = load_f32x2_global(inputs.kg.as_ptr(), compact);
            let (g_last0, g_last1) = load_f32x2_global(inputs.g.as_ptr(), last);
            let beta_value =
                inputs.beta[beta_compact_index(ctx.batch, token, ctx.head, ctx.params)];
            cvt_f16x2_f32(
                beta_value * kg0 * kda_decay_exp(2.0 * g0 - g_last0),
                beta_value * kg1 * kda_decay_exp(2.0 * g1 - g_last1),
            )
        } else if token < ctx.end && dim < ctx.params.head_dim {
            let compact = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            let g_last = inputs.g[compact_index(ctx.batch, ctx.end - 1, ctx.head, dim, ctx.params)];
            let beta_value =
                inputs.beta[beta_compact_index(ctx.batch, token, ctx.head, ctx.params)];
            cvt_f16x2_f32(
                beta_value * inputs.kg[compact] * kda_decay_exp(2.0 * inputs.g[compact] - g_last),
                0.0,
            )
        } else {
            0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}
