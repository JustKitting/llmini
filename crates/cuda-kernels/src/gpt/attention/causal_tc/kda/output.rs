use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::store_f16x2_shared;
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_K};
use crate::kda_tc::{
    CtaATile, CtaTiles, KdaChunkTileCtx, MatrixTileCtx, StateTileLayout, mma_accumulate_pair,
    stage_compact_half_a, stage_compact_half_b_t as stage_vnew_b_t_slice, stage_state_b_t,
    store_hidden_output_quad, swizzle_tc_offset,
};

pub(in super::super) fn chunk_kda_output_from_state_body(
    q: &[u16],
    v_new: &[u16],
    aqk: &[u16],
    out: DisjointSlice<f32>,
    chunk_states: &[u16],
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    output_body::<false>(q, v_new, aqk, out, chunk_states, params, tiles);
}

pub(crate) fn exact_kda_output_body(
    q: &[u16],
    v_new: &[u16],
    aqk: &[u16],
    out: DisjointSlice<f32>,
    chunk_states: &[u16],
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    output_body::<true>(q, v_new, aqk, out, chunk_states, params, tiles);
}

#[inline(always)]
fn output_body<const PADDED: bool>(
    q: &[u16],
    v_new: &[u16],
    aqk: &[u16],
    mut out: DisjointSlice<f32>,
    chunk_states: &[u16],
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    let Some(mut ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    if PADDED {
        ctx.compact.seq_len = params.seq_len.div_ceil(64) * 64;
    }
    let (a_tile, b_tile) = tiles;
    let compact_ctx = ctx.compact;

    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < params.head_dim {
        stage_compact_half_a(q, a_tile, compact_ctx, k_base);
        stage_state_b_t(chunk_states, b_tile, ctx, k_base, StateTileLayout::KV);
        thread::sync_threads();
        mma_accumulate_pair(compact_ctx.tile, a_tile, b_tile, &mut acc0, &mut acc1);
        thread::sync_threads();
        k_base += CTA_K;
    }
    let mut k_base = 0;
    while k_base < params.chunk_size {
        stage_chunk_matrix_a(aqk, a_tile, ctx.matrix, k_base);
        stage_vnew_b_t_slice(v_new, b_tile, compact_ctx, k_base);
        thread::sync_threads();
        mma_accumulate_pair(compact_ctx.tile, a_tile, b_tile, &mut acc0, &mut acc1);
        thread::sync_threads();
        k_base += CTA_K;
    }

    let mut output_ctx = compact_ctx;
    if PADDED {
        output_ctx.seq_len = params.seq_len;
    }
    store_hidden_output_quad(acc0, &mut out, output_ctx, compact_ctx.tile.warp_n0);
    store_hidden_output_quad(acc1, &mut out, output_ctx, compact_ctx.tile.warp_n0 + 1);
}

fn stage_chunk_matrix_a(src: &[u16], a_tile: &mut CtaATile, ctx: MatrixTileCtx, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token_in_chunk = ctx.tile.row_base + row;
        let source = k_base + col;
        let packed = if token_in_chunk < ctx.chunk_size && source + 1 < ctx.chunk_size {
            let index = ctx.index(token_in_chunk, source);
            crate::f16_tc_matmul::convert::load_f16x2_global_bits_read_only(src.as_ptr(), index)
        } else if token_in_chunk < ctx.chunk_size && source < ctx.chunk_size {
            let index = ctx.index(token_in_chunk, source);
            u32::from(
                crate::f16_tc_matmul::convert::load_f16_global_bits_read_only(src.as_ptr(), index),
            )
        } else {
            0
        };
        store_f16x2_shared(
            a_tile.as_mut_ptr(),
            swizzle_tc_offset(pair) as usize,
            packed,
        );
        pair += thread::blockDim_x() * 2;
    }
}
