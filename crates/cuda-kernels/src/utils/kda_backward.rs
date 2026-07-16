use cuda_device::{convert::cvt_f16x2_f32, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{
    load_f16x2_global, load_f32x2_shared, store_f16x2_shared, store_f32x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::{chunk_state_index, compact_index, hidden_index, kda_decay_exp};
use crate::kda_tc::{
    CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, compact_fragment_coords,
    for_acc_fragment_pairs,
};

pub(crate) fn stage_compact_t_a(
    src: &[f32],
    a_tile: &mut CtaATile,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
    scale: f32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let dim = ctx.tile.row_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if dim < ctx.params.head_dim && token0 < ctx.end {
            scale * src[compact_index(ctx.batch, token0, ctx.head, dim, ctx.params)]
        } else {
            0.0
        };
        let hi = if dim < ctx.params.head_dim && token1 < ctx.end {
            scale * src[compact_index(ctx.batch, token1, ctx.head, dim, ctx.params)]
        } else {
            0.0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += CTA_THREADS * 2;
    }
}

pub(crate) fn stage_hidden_dout_b_t(
    d_out: &[f32],
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if v_dim < ctx.params.head_dim && token0 < ctx.end {
            d_out[hidden_index(ctx.batch, token0, ctx.head, v_dim, ctx.params)]
        } else {
            0.0
        };
        let hi = if v_dim < ctx.params.head_dim && token1 < ctx.end {
            d_out[hidden_index(ctx.batch, token1, ctx.head, v_dim, ctx.params)]
        } else {
            0.0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += CTA_THREADS * 2;
    }
}

pub(crate) fn load_chunk_state(
    chunk_states: &[u16],
    state: &mut KdaStateTile,
    bh: u32,
    chunk: u32,
    state_elems: u32,
    params: &CausalAttentionParams,
    threads_per_block: u32,
) {
    let state_base = chunk_state_index(bh, chunk, 0, params);
    let mut pair = thread::threadIdx_x() * 2;
    while pair < state_elems {
        let (lo, hi) = load_f16x2_global(chunk_states.as_ptr(), state_base + pair as usize);
        state[pair as usize] = lo;
        state[pair as usize + 1] = hi;
        pair += threads_per_block * 2;
    }
    thread::sync_threads();
}

pub(crate) fn store_dh_quads(
    acc: [[f32; 4]; 4],
    d_h_next: &KdaStateTile,
    d_h: &mut KdaStateTile,
    g: &[f32],
    ctx: CompactTileCtx<'_>,
) {
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (k_dim0, v_dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (k_dim1, v_dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        if k_dim0 < ctx.params.head_dim
            && k_dim1 == k_dim0
            && v_dim0 + 1 == v_dim1
            && v_dim1 < ctx.params.head_dim
        {
            let index = (k_dim0 * ctx.params.head_dim + v_dim0) as usize;
            let g_last = g[compact_index(ctx.batch, ctx.end - 1, ctx.head, k_dim0, ctx.params)];
            let decay = kda_decay_exp(g_last);
            let (next_lo, next_hi) = load_f32x2_shared(d_h_next.as_ptr(), index);
            store_f32x2_shared(
                d_h.as_mut_ptr(),
                index,
                decay * next_lo + lo,
                decay * next_hi + hi,
            );
        } else {
            if k_dim0 < ctx.params.head_dim && v_dim0 < ctx.params.head_dim {
                let index = (k_dim0 * ctx.params.head_dim + v_dim0) as usize;
                let g_last = g[compact_index(ctx.batch, ctx.end - 1, ctx.head, k_dim0, ctx.params)];
                d_h[index] = kda_decay_exp(g_last) * d_h_next[index] + lo;
            }
            if k_dim1 < ctx.params.head_dim && v_dim1 < ctx.params.head_dim {
                let index = (k_dim1 * ctx.params.head_dim + v_dim1) as usize;
                let g_last = g[compact_index(ctx.batch, ctx.end - 1, ctx.head, k_dim1, ctx.params)];
                d_h[index] = kda_decay_exp(g_last) * d_h_next[index] + hi;
            }
        }
    });
}
