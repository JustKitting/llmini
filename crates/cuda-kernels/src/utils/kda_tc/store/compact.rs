use cuda_device::DisjointSlice;

use super::coords::compact_fragment_coords;
use crate::f16_tc_matmul::convert::{load_f32x2_global, store_f32x2_global};
use crate::kda_common::{compact_index, hidden_index};
use crate::kda_tc::CompactTileCtx;

pub(crate) fn store_vnew_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    u: &[f32],
    v_new: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
) {
    let out_ptr = v_new.as_mut_ptr();
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (token_in_chunk0, v_dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (token_in_chunk1, v_dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + token_in_chunk0;
        let token1 = ctx.start + token_in_chunk1;
        if token0 < ctx.end
            && token1 == token0
            && v_dim0 + 1 == v_dim1
            && v_dim1 < ctx.params.head_dim
        {
            let index = compact_index(ctx.batch, token0, ctx.head, v_dim0, ctx.params);
            let (u_lo, u_hi) = load_f32x2_global(u.as_ptr(), index);
            store_f32x2_global(out_ptr, index, u_lo - lo, u_hi - hi);
        } else {
            if token0 < ctx.end && v_dim0 < ctx.params.head_dim {
                let index = compact_index(ctx.batch, token0, ctx.head, v_dim0, ctx.params);
                unsafe {
                    *v_new.get_unchecked_mut(index) = u[index] - lo;
                }
            }
            if token1 < ctx.end && v_dim1 < ctx.params.head_dim {
                let index = compact_index(ctx.batch, token1, ctx.head, v_dim1, ctx.params);
                unsafe {
                    *v_new.get_unchecked_mut(index) = u[index] - hi;
                }
            }
        }
    });
}

#[derive(Clone, Copy)]
pub(crate) enum CompactStore {
    SetScaled(f32),
    Add,
}

pub(crate) fn store_compact_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    dst: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
    mode: CompactStore,
) {
    let dst_ptr = dst.as_mut_ptr();
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (token_in_chunk0, dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (token_in_chunk1, dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + token_in_chunk0;
        let token1 = ctx.start + token_in_chunk1;
        if token0 < ctx.end && token1 == token0 && dim0 + 1 == dim1 && dim1 < ctx.params.head_dim {
            let index = compact_index(ctx.batch, token0, ctx.head, dim0, ctx.params);
            match mode {
                CompactStore::SetScaled(scale) => {
                    store_f32x2_global(dst_ptr, index, lo * scale, hi * scale);
                }
                CompactStore::Add => {
                    let (dst_lo, dst_hi) = load_f32x2_global(dst_ptr, index);
                    store_f32x2_global(dst_ptr, index, dst_lo + lo, dst_hi + hi);
                }
            }
        } else {
            if token0 < ctx.end && dim0 < ctx.params.head_dim {
                let index = compact_index(ctx.batch, token0, ctx.head, dim0, ctx.params);
                unsafe {
                    match mode {
                        CompactStore::SetScaled(scale) => {
                            *dst.get_unchecked_mut(index) = lo * scale;
                        }
                        CompactStore::Add => *dst.get_unchecked_mut(index) += lo,
                    }
                }
            }
            if token1 < ctx.end && dim1 < ctx.params.head_dim {
                let index = compact_index(ctx.batch, token1, ctx.head, dim1, ctx.params);
                unsafe {
                    match mode {
                        CompactStore::SetScaled(scale) => {
                            *dst.get_unchecked_mut(index) = hi * scale;
                        }
                        CompactStore::Add => *dst.get_unchecked_mut(index) += hi,
                    }
                }
            }
        }
    });
}

pub(crate) fn store_hidden_output_quads<const N_REPEATS: usize>(
    acc: [[f32; 4]; N_REPEATS],
    out: &mut DisjointSlice<f32>,
    ctx: CompactTileCtx<'_>,
) {
    let out_ptr = out.as_mut_ptr();
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (token_in_chunk0, dim0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (token_in_chunk1, dim1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        let token0 = ctx.start + token_in_chunk0;
        let token1 = ctx.start + token_in_chunk1;
        let row0 = ctx.batch * ctx.params.seq_len + token0;
        let row1 = ctx.batch * ctx.params.seq_len + token1;
        if token0 < ctx.end
            && token1 == token0
            && row0 < ctx.params.row_count
            && row1 == row0
            && dim0 + 1 == dim1
            && dim1 < ctx.params.head_dim
        {
            let index = hidden_index(ctx.batch, token0, ctx.head, dim0, ctx.params);
            store_f32x2_global(out_ptr, index, lo, hi);
        } else {
            if token0 < ctx.end && row0 < ctx.params.row_count && dim0 < ctx.params.head_dim {
                let index = hidden_index(ctx.batch, token0, ctx.head, dim0, ctx.params);
                unsafe {
                    *out.get_unchecked_mut(index) = lo;
                }
            }
            if token1 < ctx.end && row1 < ctx.params.row_count && dim1 < ctx.params.head_dim {
                let index = hidden_index(ctx.batch, token1, ctx.head, dim1, ctx.params);
                unsafe {
                    *out.get_unchecked_mut(index) = hi;
                }
            }
        }
    });
}
