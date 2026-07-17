use cuda_device::{DisjointSlice, SharedArray, convert::cvt_f16x2_f32, thread};

use crate::f16_tc_matmul::convert::{
    load_f32_global_read_only, load_f32x2_global_read_only, store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K};
use crate::kda_common::compact_index;
use crate::kda_tc::CompactTileCtx;

pub(crate) fn stage_compact_a(
    src: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token = ctx.start + ctx.tile.row_base + row;
        let dim = k_base + col;
        let packed = if token < ctx.end && dim + 1 < ctx.params.head_dim {
            let index = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else {
            0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

pub(crate) fn stage_compact_b_t(
    src: &[f32],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
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
                src.as_ptr(),
                compact_index(ctx.batch, token0, ctx.head, v_dim, ctx.params),
            )
        } else {
            0.0
        };
        let hi = if v_dim < ctx.params.head_dim && token1 < ctx.end {
            load_f32_global_read_only(
                src.as_ptr(),
                compact_index(ctx.batch, token1, ctx.head, v_dim, ctx.params),
            )
        } else {
            0.0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += thread::blockDim_x() * 2;
    }
}

pub(crate) fn stage_compact_token_dim_b_t(
    src: &[f32],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx<'_>,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token = ctx.start + ctx.tile.col_base + row;
        let dim = k_base + col;
        let packed = if token < ctx.end && dim + 1 < ctx.params.head_dim {
            let index = compact_index(ctx.batch, token, ctx.head, dim, ctx.params);
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else if token < ctx.end && dim < ctx.params.head_dim {
            cvt_f16x2_f32(
                load_f32_global_read_only(
                    src.as_ptr(),
                    compact_index(ctx.batch, token, ctx.head, dim, ctx.params),
                ),
                0.0,
            )
        } else {
            0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, packed);
        pair += thread::blockDim_x() * 2;
    }
}

pub(crate) fn stage_compact_b_t_disjoint(
    src: &mut DisjointSlice<f32>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
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
            let index = compact_index(ctx.batch, token0, ctx.head, v_dim, ctx.params);
            unsafe { *src.get_unchecked_mut(index) }
        } else {
            0.0
        };
        let hi = if v_dim < ctx.params.head_dim && token1 < ctx.end {
            let index = compact_index(ctx.batch, token1, ctx.head, v_dim, ctx.params);
            unsafe { *src.get_unchecked_mut(index) }
        } else {
            0.0
        };
        store_f16x2_shared(b_tile.as_mut_ptr(), pair as usize, cvt_f16x2_f32(lo, hi));
        pair += thread::blockDim_x() * 2;
    }
}
