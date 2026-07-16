use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::kda_tc::{
    CompactStore::SetScaled, CtaTiles, KdaChunkTileCtx, StateTileLayout, StateTileSource,
    mma_accumulate, stage_compact_a as stage_dm_compact_a, stage_state_b_t, store_compact_quads,
    store_vnew_quads, tc_stage_loop,
};

pub(crate) fn chunk_kda_dkg_from_vnew_dh_body(
    v_new: &[f32],
    d_h_states: &[f32],
    mut d_kg: DisjointSlice<f32>,
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
        stage_dm_compact_a(v_new, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(StateTileSource::F32(d_h_states), b_tile, ctx, k_base, StateTileLayout::VK);
    });

    store_compact_quads(acc, &mut d_kg, compact_ctx, SetScaled(1.0));
}

#[derive(Clone, Copy)]
pub(crate) enum ChunkStateMatmulMode {
    VNew,
    Dw,
    Dqg,
}

pub(crate) fn chunk_state_matmul_body(
    a_src: &[f32],
    vnew_base: &[f32],
    state_u16: &[u16],
    mut out: DisjointSlice<f32>,
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
    mode: ChunkStateMatmulMode,
) {
    let Some(ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    let (a_tile, b_tile) = tiles;
    let compact_ctx = ctx.compact;

    let mut acc = [[0.0_f32; 4]; 2];
    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.head_dim; {
        stage_dm_compact_a(a_src, a_tile, compact_ctx, k_base);
    } {
        match mode {
            ChunkStateMatmulMode::VNew => {
                stage_state_b_t(StateTileSource::F16(state_u16), b_tile, ctx, k_base, StateTileLayout::KV)
            }
            ChunkStateMatmulMode::Dw | ChunkStateMatmulMode::Dqg => {
                stage_state_b_t(StateTileSource::F16(state_u16), b_tile, ctx, k_base, StateTileLayout::VK)
            },
        }
    });

    match mode {
        ChunkStateMatmulMode::VNew => store_vnew_quads(acc, vnew_base, &mut out, compact_ctx),
        ChunkStateMatmulMode::Dw => {
            store_compact_quads(acc, &mut out, compact_ctx, SetScaled(-1.0))
        }
        ChunkStateMatmulMode::Dqg => {
            store_compact_quads(acc, &mut out, compact_ctx, SetScaled(1.0))
        }
    }
}

pub(crate) fn chunk_state_dw_dqg_matmul_body(
    d_u: &[f32],
    d_out_compact: &[f32],
    state_u16: &[u16],
    mut d_w: DisjointSlice<f32>,
    mut d_qg: DisjointSlice<f32>,
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    let Some(ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    let (a_tile, b_tile) = tiles;
    let compact_ctx = ctx.compact;

    let mut dw_acc = [[0.0_f32; 4]; 2];
    let mut dqg_acc = [[0.0_f32; 4]; 2];
    let mut k_base = 0;
    while k_base < params.head_dim {
        stage_state_b_t(
            StateTileSource::F16(state_u16),
            b_tile,
            ctx,
            k_base,
            StateTileLayout::VK,
        );
        stage_dm_compact_a(d_u, a_tile, compact_ctx, k_base);
        thread::sync_threads();
        mma_accumulate(compact_ctx.tile, a_tile, b_tile, &mut dw_acc);
        thread::sync_threads();

        stage_dm_compact_a(d_out_compact, a_tile, compact_ctx, k_base);
        thread::sync_threads();
        mma_accumulate(compact_ctx.tile, a_tile, b_tile, &mut dqg_acc);
        thread::sync_threads();
        k_base += crate::f16_tc_matmul::cta_tile::CTA_K;
    }

    store_compact_quads(dw_acc, &mut d_w, compact_ctx, SetScaled(-1.0));
    store_compact_quads(dqg_acc, &mut d_qg, compact_ctx, SetScaled(1.0));
}
