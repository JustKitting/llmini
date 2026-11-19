use cuda_device::{SharedArray, convert::cvt_f16x2_f32, thread};

use crate::f16_tc_matmul::convert::{
    load_f16_global_bits_read_only, load_f16x2_global_bits_read_only, store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_B_ELEMS, CTA_K};
use crate::kda_tc::{CompactTileCtx, KdaChunkTileCtx, swizzle_tc_offset};

pub(crate) fn stage_shared_state_b_t<const STATE_ELEMS: usize>(
    state: &SharedArray<f32, STATE_ELEMS>,
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: CompactTileCtx,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let row = pair / CTA_K;
        let col = pair - row * CTA_K;
        let v_dim = ctx.tile.col_base + row;
        let k_dim = k_base + col;
        let packed = if v_dim < ctx.head_dim && k_dim + 1 < ctx.head_dim {
            cvt_f16x2_f32(
                state[(k_dim * ctx.head_dim + v_dim) as usize],
                state[((k_dim + 1) * ctx.head_dim + v_dim) as usize],
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

#[derive(Clone, Copy)]
pub(crate) enum StateTileLayout {
    KV,
    VK,
}

pub(crate) fn stage_state_b_t(
    states: &[u16],
    b_tile: &mut SharedArray<u16, CTA_B_ELEMS>,
    ctx: KdaChunkTileCtx,
    k_base: u32,
    layout: StateTileLayout,
) {
    if matches!(layout, StateTileLayout::VK) {
        let mut pair = thread::threadIdx_x() * 2;
        while pair < CTA_B_ELEMS as u32 {
            let row = pair / CTA_K;
            let col = pair - row * CTA_K;
            let k_dim = ctx.compact.tile.col_base + row;
            let v_dim = k_base + col;
            let packed = if k_dim < ctx.compact.head_dim && v_dim + 1 < ctx.compact.head_dim {
                let index = ctx.state_index(k_dim * ctx.compact.head_dim + v_dim);
                load_f16x2_global_bits_read_only(states.as_ptr(), index)
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
        return;
    }

    let mut offset = thread::threadIdx_x();
    while offset < CTA_B_ELEMS as u32 {
        let row = offset / CTA_K;
        let col = offset - row * CTA_K;
        let (k_dim, v_dim) = match layout {
            StateTileLayout::KV => (k_base + col, ctx.compact.tile.col_base + row),
            StateTileLayout::VK => (ctx.compact.tile.col_base + row, k_base + col),
        };
        b_tile[swizzle_tc_offset(offset) as usize] =
            if k_dim < ctx.compact.head_dim && v_dim < ctx.compact.head_dim {
                let index = ctx.state_index(k_dim * ctx.compact.head_dim + v_dim);
                load_f16_global_bits_read_only(states.as_ptr(), index)
            } else {
                0
            };
        offset += thread::blockDim_x();
    }
}
