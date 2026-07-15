use cuda_device::thread;

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::{chunk_state_index, compact_index, hidden_index, kda_decay_exp};
use crate::kda_tc::{
    CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, compact_fragment_coords, for_acc_fragments,
};

macro_rules! stage_compact_t_a_fn {
    ($name:ident, $src:ident: $src_ty:ty, $index:ident, $read:expr) => {
        pub(crate) fn $name(
            $src: $src_ty,
            a_tile: &mut CtaATile,
            ctx: CompactTileCtx<'_>,
            k_base: u32,
            scale: f32,
        ) {
            let mut offset = thread::threadIdx_x();
            while offset < CTA_A_ELEMS as u32 {
                let row = offset / CTA_K;
                let col = offset - row * CTA_K;
                let dim = ctx.tile.row_base + row;
                let token = ctx.start + k_base + col;
                a_tile[offset as usize] = if dim < ctx.params.head_dim && token < ctx.end {
                    let $index = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
                    cvt_rn_f16_f32(scale * $read)
                } else {
                    0
                };
                offset += CTA_THREADS;
            }
        }
    };
}

stage_compact_t_a_fn!(stage_compact_t_a, src: &[f32], index, src[index]);

pub(crate) fn stage_hidden_dout_b_t(
    d_out: &[f32],
    b_tile: &mut CtaBTile,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let token = ctx.start + k_base + col;
        b_tile[offset as usize] = if v_dim < ctx.params.head_dim && token < ctx.end {
            cvt_rn_f16_f32(d_out[hidden_index(ctx.batch, token, ctx.head, v_dim, ctx.params)])
        } else {
            0
        };
        offset += CTA_THREADS;
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
    let mut idx = thread::threadIdx_x();
    while idx < state_elems {
        state[idx as usize] = cvt_f32_f16(chunk_states[chunk_state_index(bh, chunk, idx, params)]);
        idx += threads_per_block;
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
    for_acc_fragments!(acc, ctx.tile, |warp_n, frag, value| {
        let (k_dim, v_dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        if k_dim < ctx.params.head_dim && v_dim < ctx.params.head_dim {
            let index = (k_dim * ctx.params.head_dim + v_dim) as usize;
            let g_last = g[compact_index(ctx.batch, ctx.end - 1, ctx.head, k_dim, ctx.params)];
            d_h[index] = kda_decay_exp(g_last) * d_h_next[index] + value;
        }
    });
}
