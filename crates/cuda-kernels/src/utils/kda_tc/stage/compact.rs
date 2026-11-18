use cuda_device::{DisjointSlice, SharedArray, convert::cvt_f16x2_f32, thread};

use crate::f16_tc_matmul::convert::{
    load_f32x2_global_read_only, store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K};
use crate::kda_tc::{CompactTileCtx, swizzle_tc_offset};

pub(crate) fn stage_compact_a(
    src: &[f32],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: CompactTileCtx,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token = ctx.start + ctx.tile.row_base + row;
        let dim = k_base + col;
        let packed = if token < ctx.end && dim + 1 < ctx.head_dim {
            let index = ctx.compact_index(token, dim);
            let (lo, hi) = load_f32x2_global_read_only(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
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

pub(crate) fn stage_compact_half_b_t_disjoint(
    src: &mut DisjointSlice<u16>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if v_dim < ctx.head_dim && token0 < ctx.end {
            let index = ctx.compact_index(token0, v_dim);
            unsafe { *src.get_unchecked_mut(index) }
        } else {
            0
        };
        let hi = if v_dim < ctx.head_dim && token1 < ctx.end {
            let index = ctx.compact_index(token1, v_dim);
            unsafe { *src.get_unchecked_mut(index) }
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

pub(crate) fn stage_compact_half_a(
    src: &[u16],
    a_tile: &mut SharedArray<u16, CTA_A_ELEMS>,
    ctx: CompactTileCtx,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_A_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let token = ctx.start + ctx.tile.row_base + row;
        let dim = k_base + col;
        let packed = if token < ctx.end && dim + 1 < ctx.head_dim {
            let index = ctx.compact_index(token, dim);
            crate::f16_tc_matmul::convert::load_f16x2_global_bits_read_only(src.as_ptr(), index)
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

pub(crate) fn stage_compact_half_b_t(
    src: &[u16],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let token0 = ctx.start + k_base + col;
        let token1 = token0 + 1;
        let lo = if v_dim < ctx.head_dim && token0 < ctx.end {
            crate::f16_tc_matmul::convert::load_f16_global_bits_read_only(
                src.as_ptr(),
                ctx.compact_index(token0, v_dim),
            )
        } else {
            0
        };
        let hi = if v_dim < ctx.head_dim && token1 < ctx.end {
            crate::f16_tc_matmul::convert::load_f16_global_bits_read_only(
                src.as_ptr(),
                ctx.compact_index(token1, v_dim),
            )
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
