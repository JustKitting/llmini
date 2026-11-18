use crate::f16_tc_matmul::convert::{
    load_f16_global_bits_read_only, store_f16x2_global, store_f16x2_shared,
};
use crate::f16_tc_matmul::cta_tile::{CTA_A_ELEMS, CTA_B_ELEMS, CTA_K};
use crate::kda_tc::{CompactTileCtx, CtaATile, compact_fragment_coords, swizzle_tc_offset};
use cuda_device::{DisjointSlice, SharedArray, convert::cvt_f16x2_f32, thread};

use crate::attention::CausalAttentionParams;
use crate::kda_tc::{
    CtaTiles, KdaChunkTileCtx, StateTileLayout, mma_accumulate_pair, stage_state_b_t,
    store_compact_half_quad, store_compact_quad, store_compact_quads, store_vnew_quads,
    tc_stage_loop,
};

pub(crate) fn chunk_kda_dkg_from_vnew_dh_body(
    v_new: &[u16],
    d_h_states: &[u16],
    mut d_kg: DisjointSlice<f32>,
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
        crate::kda_tc::stage_compact_half_a(v_new, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(d_h_states, b_tile, ctx, k_base, StateTileLayout::VK);
    });

    store_compact_quads(acc, &mut d_kg, compact_ctx, 1.0);
}

const NATIVE_STATE_TILE_STRIDE: u32 = 66;
pub(crate) const NATIVE_STATE_TILE_ELEMS: usize = 64 * NATIVE_STATE_TILE_STRIDE as usize;
type NativeTile = SharedArray<u16, NATIVE_STATE_TILE_ELEMS>;

// Both locations contain the same native FP16 representation. Callers select
// a location statically; there is no alternate precision or runtime format.
#[derive(Clone, Copy)]
enum NativeSource<'a> {
    Global(&'a [u16]),
    Shared(&'a NativeTile),
}
impl NativeSource<'_> {
    #[inline(always)]
    fn stage(self, dst: &mut CtaATile, ctx: CompactTileCtx, k: u32) {
        match self {
            Self::Global(values) => crate::kda_tc::stage_compact_half_a(values, dst, ctx, k),
            Self::Shared(values) => {
                let mut pair = thread::threadIdx_x() * 2;
                while pair < CTA_A_ELEMS as u32 {
                    let row = ctx.tile.row_base + pair / CTA_K;
                    let col = k + pair % CTA_K;
                    let bits = if ctx.start + row < ctx.end && col + 1 < ctx.head_dim {
                        unsafe {
                            *(values
                                .as_ptr()
                                .add((row * NATIVE_STATE_TILE_STRIDE + col) as usize)
                                as *const u32)
                        }
                    } else {
                        0
                    };
                    store_f16x2_shared(dst.as_mut_ptr(), swizzle_tc_offset(pair) as usize, bits);
                    pair += thread::blockDim_x() * 2;
                }
            }
        }
    }
}

pub(crate) fn chunk_state_dw_dqg_matmul_body(
    d_u: &[u16],
    d_out: &[u16],
    state: &[u16],
    d_w: DisjointSlice<u16>,
    d_qg: DisjointSlice<f32>,
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
) {
    let Some(ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    dw_dqg_from_sources(
        NativeSource::Global(d_u),
        NativeSource::Global(d_out),
        state,
        d_w,
        d_qg,
        params,
        ctx,
        tiles,
    );
}

#[inline(always)]
fn dw_dqg_from_sources(
    d_u: NativeSource<'_>,
    d_out: NativeSource<'_>,
    state_u16: &[u16],
    mut d_w: DisjointSlice<u16>,
    mut d_qg: DisjointSlice<f32>,
    params: CausalAttentionParams,
    ctx: KdaChunkTileCtx,
    tiles: CtaTiles<'_>,
) {
    let (a_tile, b_tile) = tiles;
    let compact_ctx = ctx.compact;

    let mut dw_acc0 = [0.0_f32; 4];
    let mut dw_acc1 = [0.0_f32; 4];
    let mut dqg_acc0 = [0.0_f32; 4];
    let mut dqg_acc1 = [0.0_f32; 4];
    let mut k_base = 0;
    while k_base < params.head_dim {
        stage_state_b_t(state_u16, b_tile, ctx, k_base, StateTileLayout::VK);
        d_u.stage(a_tile, compact_ctx, k_base);
        thread::sync_threads();
        mma_accumulate_pair(compact_ctx.tile, a_tile, b_tile, &mut dw_acc0, &mut dw_acc1);
        thread::sync_threads();

        d_out.stage(a_tile, compact_ctx, k_base);
        thread::sync_threads();
        mma_accumulate_pair(
            compact_ctx.tile,
            a_tile,
            b_tile,
            &mut dqg_acc0,
            &mut dqg_acc1,
        );
        thread::sync_threads();
        k_base += crate::f16_tc_matmul::cta_tile::CTA_K;
    }

    store_compact_half_quad(
        dw_acc0,
        &mut d_w,
        compact_ctx,
        compact_ctx.tile.warp_n0,
        -1.0,
    );
    store_compact_half_quad(
        dw_acc1,
        &mut d_w,
        compact_ctx,
        compact_ctx.tile.warp_n0 + 1,
        -1.0,
    );
    store_compact_quad(
        dqg_acc0,
        &mut d_qg,
        compact_ctx,
        compact_ctx.tile.warp_n0,
        1.0,
    );
    store_compact_quad(
        dqg_acc1,
        &mut d_qg,
        compact_ctx,
        compact_ctx.tile.warp_n0 + 1,
        1.0,
    );
}

/// Reconstruct directly in the format used by every downstream tensor core.
pub(crate) fn chunk_state_vnew_body(
    w: &[u16],
    u: &[f32],
    state_u16: &[u16],
    mut out: DisjointSlice<u16>,
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
        crate::kda_tc::stage_compact_half_a(w, a_tile, compact_ctx, k_base);
    } {
        stage_state_b_t(state_u16, b_tile, ctx, k_base, StateTileLayout::KV);
    });
    store_vnew_quads(acc, u, &mut out, compact_ctx);
}

pub(crate) fn chunk_state_project_du_dw_dqg_body(
    aqk: &[u16],
    d_out: &[u16],
    state: &[u16],
    mut d_u: DisjointSlice<u16>,
    d_w: DisjointSlice<u16>,
    d_qg: DisjointSlice<f32>,
    params: CausalAttentionParams,
    tiles: CtaTiles<'_>,
    du_tile: &mut NativeTile,
    dout_tile: &mut NativeTile,
) {
    let Some(ctx) = KdaChunkTileCtx::from_wide_block(&params) else {
        return;
    };
    let (a, b) = tiles;
    let c = ctx.compact;
    let mut acc0 = [0.0_f32; 4];
    let mut acc1 = [0.0_f32; 4];
    let mut k = 0;
    while k < params.chunk_size {
        let mut pair = thread::threadIdx_x() * 2;
        while pair < CTA_A_ELEMS as u32 {
            let row = pair / CTA_K;
            let col = pair % CTA_K;
            let query = k + col;
            let key = c.tile.row_base + row;
            let lo = if query < ctx.matrix.chunk_tokens && key < ctx.matrix.chunk_tokens {
                load_f16_global_bits_read_only(aqk.as_ptr(), ctx.matrix.index(query, key))
            } else {
                0
            };
            let hi = if query + 1 < ctx.matrix.chunk_tokens && key < ctx.matrix.chunk_tokens {
                load_f16_global_bits_read_only(aqk.as_ptr(), ctx.matrix.index(query + 1, key))
            } else {
                0
            };
            store_f16x2_shared(
                a.as_mut_ptr(),
                swizzle_tc_offset(pair) as usize,
                u32::from(lo) | (u32::from(hi) << 16),
            );
            pair += thread::blockDim_x() * 2;
        }
        let mut pair = thread::threadIdx_x() * 2;
        while pair < CTA_B_ELEMS as u32 {
            let row = pair / CTA_K;
            let col = pair % CTA_K;
            let dim = c.tile.col_base + row;
            let token = c.start + k + col;
            let lo = if dim < c.head_dim && token < c.end {
                load_f16_global_bits_read_only(d_out.as_ptr(), c.compact_index(token, dim))
            } else {
                0
            };
            let hi = if dim < c.head_dim && token + 1 < c.end {
                load_f16_global_bits_read_only(d_out.as_ptr(), c.compact_index(token + 1, dim))
            } else {
                0
            };
            if dim < c.head_dim && token < c.end {
                dout_tile[((k + col) * NATIVE_STATE_TILE_STRIDE + dim) as usize] = lo;
            }
            if dim < c.head_dim && token + 1 < c.end {
                dout_tile[((k + col + 1) * NATIVE_STATE_TILE_STRIDE + dim) as usize] = hi;
            }
            store_f16x2_shared(
                b.as_mut_ptr(),
                swizzle_tc_offset(pair) as usize,
                u32::from(lo) | (u32::from(hi) << 16),
            );
            pair += thread::blockDim_x() * 2;
        }
        thread::sync_threads();
        mma_accumulate_pair(c.tile, a, b, &mut acc0, &mut acc1);
        thread::sync_threads();
        k += CTA_K;
    }
    store_projected_du(acc0, &mut d_u, du_tile, c, c.tile.warp_n0);
    store_projected_du(acc1, &mut d_u, du_tile, c, c.tile.warp_n0 + 1);
    thread::sync_threads();
    dw_dqg_from_sources(
        NativeSource::Shared(du_tile),
        NativeSource::Shared(dout_tile),
        state,
        d_w,
        d_qg,
        params,
        ctx,
        (a, b),
    );
}

#[inline(always)]
fn store_projected_du(
    acc: [f32; 4],
    output: &mut DisjointSlice<u16>,
    shared: &mut NativeTile,
    ctx: CompactTileCtx,
    warp_n: u32,
) {
    let mut frag = 0;
    while frag < 4 {
        let (row, dim) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let token = ctx.start + row;
        if token < ctx.end && dim + 1 < ctx.head_dim {
            let bits = cvt_f16x2_f32(acc[frag], acc[frag + 1]);
            // One rounding supplies both the native base and the local MMA.
            store_f16x2_global(output.as_mut_ptr(), ctx.compact_index(token, dim), bits);
            store_f16x2_shared(
                shared.as_mut_ptr(),
                (row * NATIVE_STATE_TILE_STRIDE + dim) as usize,
                bits,
            );
        }
        frag += 2;
    }
}
