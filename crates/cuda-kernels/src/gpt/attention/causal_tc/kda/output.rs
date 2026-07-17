use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{
    load_f32_global_read_only, load_f32x2_global_read_only, store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_K};
use crate::kda_common::chunk_matrix_index;
use crate::kda_tc::{
    CtaATile, CtaTiles, KdaChunkTileCtx, MatrixTileCtx, StateTileLayout, StateTileSource,
    stage_compact_a, stage_compact_b_t as stage_vnew_b_t_slice, stage_state_b_t,
    store_hidden_output_quads, tc_stage_loop,
};

pub(in super::super) fn chunk_kda_output_from_state_body(
    q: &[f32],
    v_new: &[f32],
    aqk: &[f32],
    mut out: DisjointSlice<f32>,
    chunk_states: &[u16],
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
        stage_compact_a(q, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(StateTileSource::F16(chunk_states), b_tile, ctx, k_base, StateTileLayout::KV);
    });
    tc_stage_loop!(compact_ctx.tile, a_tile, b_tile, acc; k_base < params.chunk_size; {
        stage_chunk_matrix_a(aqk, a_tile, ctx.matrix, k_base);
    } {
        stage_vnew_b_t_slice(v_new, b_tile, compact_ctx, k_base);
    });

    store_hidden_output_quads(acc, &mut out, compact_ctx);
}

fn stage_chunk_matrix_a(src: &[f32], a_tile: &mut CtaATile, ctx: MatrixTileCtx<'_>, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token_in_chunk = ctx.tile.row_base + row;
        let source = k_base + col;
        let packed = if token_in_chunk < ctx.params.chunk_size && source + 1 < ctx.params.chunk_size
        {
            let index = chunk_matrix_index(ctx.bh, ctx.chunk, token_in_chunk, source, ctx.params);
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else if token_in_chunk < ctx.params.chunk_size && source < ctx.params.chunk_size {
            let index = chunk_matrix_index(ctx.bh, ctx.chunk, token_in_chunk, source, ctx.params);
            cvt_f16x2_f32(load_f32_global_read_only(src.as_ptr(), index), 0.0)
        } else {
            0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}
