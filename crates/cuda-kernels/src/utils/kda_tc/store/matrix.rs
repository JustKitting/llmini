use cuda_device::DisjointSlice;

use super::coords::compact_fragment_coords;
use crate::f16_tc_matmul::convert::store_f32x2_global;
use crate::kda_common::chunk_matrix_index;
use crate::kda_tc::MatrixTileCtx;

pub(crate) fn store_chunk_matrix_quads(
    acc: [[f32; 4]; 4],
    dst: &mut DisjointSlice<f32>,
    ctx: MatrixTileCtx<'_>,
) {
    let dst_ptr = dst.as_mut_ptr();
    for_acc_fragment_pairs!(acc, ctx.tile, |warp_n, frag, lo, hi| {
        let (row0, col0) = compact_fragment_coords(ctx.tile, warp_n, frag);
        let (row1, col1) = compact_fragment_coords(ctx.tile, warp_n, frag + 1);
        if row0 < ctx.params.chunk_size
            && row1 == row0
            && col0 + 1 == col1
            && col1 < ctx.params.chunk_size
        {
            let lo = if row0 < ctx.chunk_tokens && col0 < ctx.chunk_tokens {
                lo
            } else {
                0.0
            };
            let hi = if row1 < ctx.chunk_tokens && col1 < ctx.chunk_tokens {
                hi
            } else {
                0.0
            };
            let index = chunk_matrix_index(ctx.bh, ctx.chunk, row0, col0, ctx.params);
            store_f32x2_global(dst_ptr, index, lo, hi);
        } else {
            if row0 < ctx.params.chunk_size && col0 < ctx.params.chunk_size {
                let value = if row0 < ctx.chunk_tokens && col0 < ctx.chunk_tokens {
                    lo
                } else {
                    0.0
                };
                let index = chunk_matrix_index(ctx.bh, ctx.chunk, row0, col0, ctx.params);
                unsafe {
                    *dst.get_unchecked_mut(index) = value;
                }
            }
            if row1 < ctx.params.chunk_size && col1 < ctx.params.chunk_size {
                let value = if row1 < ctx.chunk_tokens && col1 < ctx.chunk_tokens {
                    hi
                } else {
                    0.0
                };
                let index = chunk_matrix_index(ctx.bh, ctx.chunk, row1, col1, ctx.params);
                unsafe {
                    *dst.get_unchecked_mut(index) = value;
                }
            }
        }
    });
}
