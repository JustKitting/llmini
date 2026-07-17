use cuda_device::{convert::cvt_f16x2_f32, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{
    load_f16x2_global_bits_read_only, load_f32_global_read_only, load_f32x2_shared,
    store_f16x2_shared, store_f32x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K};
use crate::kda_common::{chunk_state_index, compact_index, hidden_index, kda_decay_exp};
use crate::kda_tc::{CompactTileCtx, CtaATile, CtaBTile, KdaStateTile, compact_fragment_coords};

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
            scale
                * load_f32_global_read_only(
                    src.as_ptr(),
                    compact_index(ctx.batch, token0, ctx.head, dim, ctx.params),
                )
        } else {
            0.0
        };
        let hi = if dim < ctx.params.head_dim && token1 < ctx.end {
            scale
                * load_f32_global_read_only(
                    src.as_ptr(),
                    compact_index(ctx.batch, token1, ctx.head, dim, ctx.params),
                )
        } else {
            0.0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += thread::blockDim_x() * 2;
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
            load_f32_global_read_only(
                d_out.as_ptr(),
                hidden_index(ctx.batch, token0, ctx.head, v_dim, ctx.params),
            )
        } else {
            0.0
        };
        let hi = if v_dim < ctx.params.head_dim && token1 < ctx.end {
            load_f32_global_read_only(
                d_out.as_ptr(),
                hidden_index(ctx.batch, token1, ctx.head, v_dim, ctx.params),
            )
        } else {
            0.0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += thread::blockDim_x() * 2;
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
        let packed =
            load_f16x2_global_bits_read_only(chunk_states.as_ptr(), state_base + pair as usize);
        state[pair as usize] = crate::f16_tc_matmul::convert::cvt_f32_f16(packed as u16);
        state[pair as usize + 1] =
            crate::f16_tc_matmul::convert::cvt_f32_f16((packed >> 16) as u16);
        pair += threads_per_block * 2;
    }
    thread::sync_threads();
}

pub(crate) fn store_dh_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    d_h_next: &KdaStateTile,
    d_h: &mut KdaStateTile,
    g: &[f32],
    ctx: CompactTileCtx<'_>,
) {
    let (k_dim_0, _) = compact_fragment_coords(ctx.tile, ctx.tile.warp_n0, 0);
    let (k_dim_1, _) = compact_fragment_coords(ctx.tile, ctx.tile.warp_n0, 2);
    let last_token = ctx.end - 1;
    let decay_0 = kda_decay_exp(load_f32_global_read_only(
        g.as_ptr(),
        compact_index(ctx.batch, last_token, ctx.head, k_dim_0, ctx.params),
    ));
    let decay_1 = kda_decay_exp(load_f32_global_read_only(
        g.as_ptr(),
        compact_index(ctx.batch, last_token, ctx.head, k_dim_1, ctx.params),
    ));

    let mut i = 0;
    while i < N_REPEATS {
        let warp_n = ctx.tile.warp_n0 + i as u32;
        let (_, v_dim_0) = compact_fragment_coords(ctx.tile, warp_n, 0);
        let index_0 = (k_dim_0 * ctx.params.head_dim + v_dim_0) as usize;
        let (next_0_lo, next_0_hi) = load_f32x2_shared(d_h_next.as_ptr(), index_0);
        store_f32x2_shared(
            d_h.as_mut_ptr(),
            index_0,
            decay_0 * next_0_lo + acc[i][0],
            decay_0 * next_0_hi + acc[i][1],
        );

        let (_, v_dim_1) = compact_fragment_coords(ctx.tile, warp_n, 2);
        let index_1 = (k_dim_1 * ctx.params.head_dim + v_dim_1) as usize;
        let (next_1_lo, next_1_hi) = load_f32x2_shared(d_h_next.as_ptr(), index_1);
        store_f32x2_shared(
            d_h.as_mut_ptr(),
            index_1,
            decay_1 * next_1_lo + acc[i][2],
            decay_1 * next_1_hi + acc[i][3],
        );
        i += 1;
    }
}
