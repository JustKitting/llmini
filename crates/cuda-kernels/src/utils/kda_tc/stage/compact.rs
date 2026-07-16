use cuda_device::{DisjointSlice, SharedArray, convert::cvt_f16x2_f32, thread};

use crate::f16_tc_matmul::convert::{cvt_rn_f16_f32, load_f32x2_global, store_f16x2_shared};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K, CTA_THREADS};
use crate::kda_common::compact_index;
use crate::kda_tc::CompactTileCtx;

macro_rules! stage_compact_fn {
    ($name:ident, $src:ident: $src_ty:ty, $tile:ident: $tile_elems:ident,
     $row:ident, $col:ident, $ctx:ident, $k_base:ident, $value:expr) => {
        pub(crate) fn $name(
            $src: $src_ty,
            $tile: &mut SharedArray<u16, $tile_elems>,
            $ctx: CompactTileCtx<'_>,
            $k_base: u32,
        ) {
            let mut offset = thread::threadIdx_x();
            while offset < $tile_elems as u32 {
                let $row = offset / CTA_K;
                let $col = offset - $row * CTA_K;
                $tile[offset as usize] = $value;
                offset += CTA_THREADS;
            }
        }
    };
}

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
            let (lo, hi) = load_f32x2_global(src.as_ptr(), index);
            cvt_f16x2_f32(lo, hi)
        } else {
            0
        };
        store_f16x2_shared(a_tile.as_mut_ptr(), pair as usize, packed);
        pair += CTA_THREADS * 2;
    }
}

stage_compact_fn!(stage_compact_b_t, src: &[f32], b_tile: CTA_B_ELEMS, row, col, ctx, k_base, {
    let v_dim = ctx.tile.col_base + row;
    let token = ctx.start + k_base + col;
    if v_dim < ctx.params.head_dim && token < ctx.end {
        cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, v_dim, ctx.params)])
    } else {
        0
    }
});

stage_compact_fn!(
    stage_compact_token_dim_b_t, src: &[f32], b_tile: CTA_B_ELEMS, row, col, ctx, k_base, {
        let token = ctx.start + ctx.tile.col_base + row;
        let dim = k_base + col;
        if token < ctx.end && dim < ctx.params.head_dim {
            cvt_rn_f16_f32(src[compact_index(ctx.batch, token, ctx.head, dim, ctx.params)])
        } else {
            0
        }
    }
);

stage_compact_fn!(
    stage_compact_b_t_disjoint, src: &mut DisjointSlice<f32>, b_tile: CTA_B_ELEMS, row, col, ctx,
    k_base, {
        let v_dim = ctx.tile.col_base + row;
        let token = ctx.start + k_base + col;
        if v_dim < ctx.params.head_dim && token < ctx.end {
            let index = compact_index(ctx.batch, token, ctx.head, v_dim, ctx.params);
            cvt_rn_f16_f32(unsafe { *src.get_unchecked_mut(index) })
        } else {
            0
        }
    }
);
