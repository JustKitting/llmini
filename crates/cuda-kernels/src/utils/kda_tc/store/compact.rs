use cuda_device::DisjointSlice;

use super::coords::compact_fragment_coords;
use crate::f16_tc_matmul::convert::{load_f32x2_global, store_f32x2_global};
use crate::kda_tc::CompactTileCtx;

pub(crate) fn store_vnew_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    u: &[f32],
    v_new: &mut DisjointSlice<u16>,
    ctx: CompactTileCtx,
) {
    let out_ptr = v_new.as_mut_ptr();
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (token_in_chunk0, v_dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (token_in_chunk1, v_dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + token_in_chunk0;
        let token1 = ctx.start + token_in_chunk1;
        if token0 < ctx.end && token1 == token0 && v_dim0 + 1 == v_dim1 && v_dim1 < ctx.head_dim {
            let index = ctx.compact_index(token0, v_dim0);
            let (u_lo, u_hi) = load_f32x2_global(u.as_ptr(), index);
            crate::f16_tc_matmul::convert::store_f16x2_global(
                out_ptr,
                index,
                cuda_device::convert::cvt_f16x2_f32(u_lo - lo, u_hi - hi),
            );
        } else {
            if token0 < ctx.end && v_dim0 < ctx.head_dim {
                let index = ctx.compact_index(token0, v_dim0);
                unsafe {
                    *v_new.get_unchecked_mut(index) =
                        crate::f16_tc_matmul::convert::cvt_rn_f16_f32(u[index] - lo);
                }
            }
            if token1 < ctx.end && v_dim1 < ctx.head_dim {
                let index = ctx.compact_index(token1, v_dim1);
                unsafe {
                    *v_new.get_unchecked_mut(index) =
                        crate::f16_tc_matmul::convert::cvt_rn_f16_f32(u[index] - hi);
                }
            }
        }
    });
}

pub(crate) fn store_compact_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    dst: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx,
    scale: f32,
) {
    let mut i = 0;
    while i < N_REPEATS {
        store_compact_quad(acc[i], dst, ctx, ctx.tile.warp_n0 + i as u32, scale);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn store_compact_quad(
    acc: [f32; 4],
    dst: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx,
    warp_n: u32,
    scale: f32,
) {
    let mut frag = 0;
    while frag < 4 {
        let (row0, dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (row1, dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + row0;
        let token1 = ctx.start + row1;
        let lo = acc[frag] * scale;
        let hi = acc[frag + 1] * scale;
        if token0 < ctx.end && token1 == token0 && dim0 + 1 == dim1 && dim1 < ctx.head_dim {
            store_f32x2_global(dst.as_mut_ptr(), ctx.compact_index(token0, dim0), lo, hi);
        } else {
            if token0 < ctx.end && dim0 < ctx.head_dim {
                unsafe {
                    *dst.get_unchecked_mut(ctx.compact_index(token0, dim0)) = lo;
                }
            }
            if token1 < ctx.end && dim1 < ctx.head_dim {
                unsafe {
                    *dst.get_unchecked_mut(ctx.compact_index(token1, dim1)) = hi;
                }
            }
        }
        frag += 2;
    }
}

#[inline(always)]
pub(crate) fn store_hidden_output_quad(
    acc: [f32; 4],
    out: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx,
    warp_n: u32,
) {
    store_hidden_output_pair(acc[0], acc[1], out, ctx, warp_n, 0);
    store_hidden_output_pair(acc[2], acc[3], out, ctx, warp_n, 2);
}

#[inline(always)]
fn store_hidden_output_pair(
    lo: f32,
    hi: f32,
    out: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx,
    warp_n: u32,
    frag: usize,
) {
    let (token_in_chunk0, dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
    let (token_in_chunk1, dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
    let token0 = ctx.start + token_in_chunk0;
    let token1 = ctx.start + token_in_chunk1;
    let row0 = ctx.batch * ctx.seq_len + token0;
    let row1 = ctx.batch * ctx.seq_len + token1;
    if token0 < ctx.end
        && token1 == token0
        && row0 < ctx.row_count
        && row1 == row0
        && dim0 + 1 == dim1
        && dim1 < ctx.head_dim
    {
        let index = ctx.hidden_index(token0, dim0);
        store_f32x2_global(out.as_mut_ptr(), index, lo, hi);
        return;
    }

    if token0 < ctx.end && row0 < ctx.row_count && dim0 < ctx.head_dim {
        let index = ctx.hidden_index(token0, dim0);
        unsafe {
            *out.get_unchecked_mut(index) = lo;
        }
    }
    if token1 < ctx.end && row1 < ctx.row_count && dim1 < ctx.head_dim {
        let index = ctx.hidden_index(token1, dim1);
        unsafe {
            *out.get_unchecked_mut(index) = hi;
        }
    }
}

/// Emit the consumer's native half bits directly at the old MMA rounding point.
pub(crate) fn store_compact_half_quad(
    acc: [f32; 4],
    dst: &mut DisjointSlice<u16>,
    ctx: CompactTileCtx,
    warp_n: u32,
    scale: f32,
) {
    let mut frag = 0;
    while frag < 4 {
        let (row0, dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (row1, dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + row0;
        let token1 = ctx.start + row1;
        let lo = acc[frag] * scale;
        let hi = acc[frag + 1] * scale;
        if token0 < ctx.end && token1 == token0 && dim0 + 1 == dim1 && dim1 < ctx.head_dim {
            crate::f16_tc_matmul::convert::store_f16x2_global(
                dst.as_mut_ptr(),
                ctx.compact_index(token0, dim0),
                cuda_device::convert::cvt_f16x2_f32(lo, hi),
            );
        } else {
            if token0 < ctx.end && dim0 < ctx.head_dim {
                unsafe {
                    *dst.get_unchecked_mut(ctx.compact_index(token0, dim0)) =
                        crate::f16_tc_matmul::convert::cvt_rn_f16_f32(lo);
                }
            }
            if token1 < ctx.end && dim1 < ctx.head_dim {
                unsafe {
                    *dst.get_unchecked_mut(ctx.compact_index(token1, dim1)) =
                        crate::f16_tc_matmul::convert::cvt_rn_f16_f32(hi);
                }
            }
        }
        frag += 2;
    }
}
