use cuda_device::{convert::cvt_f16x2_f32, thread};

use super::{KDA_STATE_DIM, KDA_STATE_PARTITION_COLS, KDA_STATE_SHARED_STRIDE, KdaStateCtx};
use crate::f16_tc_matmul::convert::{
    cvt_rn_f16_f32, load_f32_global_read_only, load_f32x2_global, store_f16x2_global,
    store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_K};
use crate::kda_common::kda_decay_exp;
use crate::kda_tc::{
    CtaATile, CtaBTile, KdaDecayTile, KdaStateSaveTile, compact_fragment_coords, swizzle_tc_offset,
    tc_stage_loop,
};

#[inline(always)]
pub(super) fn compute_ws_to_vnew(
    w: *const u16,
    u: *const f32,
    v_new: *mut u16,
    vnew_shared: *mut u16,
    state: &KdaStateSaveTile,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    ctx: KdaStateCtx,
) {
    let mut acc = [[0.0_f32; 4]; 1];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < KDA_STATE_DIM; {
        stage_w_a(w, a_tile, ctx, k_base);
    } {
        stage_state_b_t(state, b_tile, ctx, k_base);
    });
    store_vnew_quads(acc, u, v_new, vnew_shared, ctx);
}

#[inline(always)]
pub(super) fn compute_kg_vnew_add_state(
    kg: *const u16,
    v_new: *mut u16,
    state: &mut KdaStateSaveTile,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    ctx: KdaStateCtx,
) {
    let mut acc = [[0.0_f32; 4]; 1];
    tc_stage_loop!(ctx.tile, a_tile, b_tile, acc; k_base < KDA_STATE_DIM; {
        stage_kg_t_a(kg, a_tile, ctx, k_base);
    } {
        stage_vnew_b_t(v_new, b_tile, ctx, k_base);
    });
    add_state_quads(acc, state, ctx);
}

#[inline(always)]
pub(super) fn decay_state(
    state: &mut KdaStateSaveTile,
    decay: &mut KdaDecayTile,
    chunk_g_last: *const f32,
    chunk: u32,
    tid: u32,
    ctx: KdaStateCtx,
) {
    if tid < KDA_STATE_DIM {
        let g_last = load_f32_global_read_only(chunk_g_last, ctx.chunk_g_last_index(chunk, tid));
        decay[tid as usize] = kda_decay_exp(g_last);
    }
    thread::sync_threads();

    let mut linear = tid;
    let partition_elems = KDA_STATE_DIM * KDA_STATE_PARTITION_COLS;
    while linear < partition_elems {
        let k_dim = linear / KDA_STATE_PARTITION_COLS;
        let v_dim = ctx.tile.col_base + linear % KDA_STATE_PARTITION_COLS;
        let state_index = k_dim * KDA_STATE_SHARED_STRIDE + v_dim;
        state[state_index as usize] *= decay[k_dim as usize];
        linear += thread::blockDim_x();
    }
}

#[inline(always)]
fn stage_w_a(src: *const u16, a_tile: &mut CtaATile, ctx: KdaStateCtx, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token = ctx.start + ctx.tile.row_base + row;
        let dim = k_base + col;
        let packed = if token < ctx.end && dim + 1 < KDA_STATE_DIM {
            let index = ctx.compact_index(token, dim);
            crate::f16_tc_matmul::convert::load_f16x2_global_bits_read_only(src, index)
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

#[inline(always)]
fn stage_state_b_t(state: &KdaStateSaveTile, b_tile: &mut CtaBTile, ctx: KdaStateCtx, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    let tile_elems = KDA_STATE_PARTITION_COLS * CTA_K;
    while pair < tile_elems {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let k_dim = k_base + col;
        let packed = if v_dim < KDA_STATE_DIM && k_dim + 1 < KDA_STATE_DIM {
            cvt_f16x2_f32(
                state[(k_dim * KDA_STATE_SHARED_STRIDE + v_dim) as usize],
                state[((k_dim + 1) * KDA_STATE_SHARED_STRIDE + v_dim) as usize],
            )
        } else {
            0
        };
        store_f16x2_shared(
            b_tile.as_mut_ptr(),
            swizzle_tc_offset(pair) as usize,
            packed,
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
fn stage_kg_t_a(src: *const u16, a_tile: &mut CtaATile, ctx: KdaStateCtx, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let dim = ctx.tile.row_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if dim < KDA_STATE_DIM && token0 < ctx.end {
            crate::f16_tc_matmul::convert::load_f16_global_bits_read_only(
                src,
                ctx.compact_index(token0, dim),
            )
        } else {
            0
        };
        let hi = if dim < KDA_STATE_DIM && token1 < ctx.end {
            crate::f16_tc_matmul::convert::load_f16_global_bits_read_only(
                src,
                ctx.compact_index(token1, dim),
            )
        } else {
            0
        };
        store_f16x2_shared(
            a_tile.as_mut_ptr(),
            swizzle_tc_offset(pair) as usize,
            u32::from(lo) | (u32::from(hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
fn stage_vnew_b_t(src: *const u16, b_tile: &mut CtaBTile, ctx: KdaStateCtx, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    let tile_elems = KDA_STATE_PARTITION_COLS * CTA_K;
    while pair < tile_elems {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if v_dim < KDA_STATE_DIM && token0 < ctx.end {
            unsafe {
                *src.add(
                    ((token0 - ctx.start) as usize) * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                        + (v_dim - ctx.tile.col_base) as usize,
                )
            }
        } else {
            0
        };
        let hi = if v_dim < KDA_STATE_DIM && token1 < ctx.end {
            unsafe {
                *src.add(
                    ((token1 - ctx.start) as usize) * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                        + (v_dim - ctx.tile.col_base) as usize,
                )
            }
        } else {
            0
        };
        store_f16x2_shared(
            b_tile.as_mut_ptr(),
            swizzle_tc_offset(pair) as usize,
            u32::from(lo) | (u32::from(hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
fn store_vnew_quads(
    acc: [[f32; 4]; 1],
    u: *const f32,
    v_new: *mut u16,
    vnew_shared: *mut u16,
    ctx: KdaStateCtx,
) {
    let warp_n = ctx.tile.warp_n0;
    let mut frag = 0;
    while frag < 4 {
        let lo = acc[0][frag];
        let hi = acc[0][frag + 1];
        let (token_in_chunk0, v_dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (token_in_chunk1, v_dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + token_in_chunk0;
        let token1 = ctx.start + token_in_chunk1;
        if token0 < ctx.end && token1 == token0 && v_dim0 + 1 == v_dim1 && v_dim1 < KDA_STATE_DIM {
            let index = ctx.compact_index(token0, v_dim0);
            let (u_lo, u_hi) = load_f32x2_global(u, index);
            let packed = cvt_f16x2_f32(u_lo - lo, u_hi - hi);
            store_f16x2_global(v_new, index, packed);
            let shared_index = token_in_chunk0 as usize * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                + (v_dim0 - ctx.tile.col_base) as usize;
            store_f16x2_shared(vnew_shared, shared_index, packed);
        } else {
            if token0 < ctx.end && v_dim0 < KDA_STATE_DIM {
                let index = ctx.compact_index(token0, v_dim0);
                unsafe {
                    let bits = cvt_rn_f16_f32(*u.add(index) - lo);
                    *v_new.add(index) = bits;
                    *vnew_shared.add(
                        token_in_chunk0 as usize * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                            + (v_dim0 - ctx.tile.col_base) as usize,
                    ) = bits;
                }
            }
            if token1 < ctx.end && v_dim1 < KDA_STATE_DIM {
                let index = ctx.compact_index(token1, v_dim1);
                unsafe {
                    let bits = cvt_rn_f16_f32(*u.add(index) - hi);
                    *v_new.add(index) = bits;
                    *vnew_shared.add(
                        token_in_chunk1 as usize * crate::kda_tc::KDA_VNEW_SHARED_STRIDE
                            + (v_dim1 - ctx.tile.col_base) as usize,
                    ) = bits;
                }
            }
        }
        frag += 2;
    }
}

#[inline(always)]
fn add_state_quads(acc: [[f32; 4]; 1], state: &mut KdaStateSaveTile, ctx: KdaStateCtx) {
    let warp_n = ctx.tile.warp_n0;
    let mut frag = 0;
    while frag < 4 {
        let (k_dim, v_dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        if k_dim < KDA_STATE_DIM && v_dim < KDA_STATE_DIM {
            state[(k_dim * KDA_STATE_SHARED_STRIDE + v_dim) as usize] += acc[0][frag];
        }
        frag += 1;
    }
}
