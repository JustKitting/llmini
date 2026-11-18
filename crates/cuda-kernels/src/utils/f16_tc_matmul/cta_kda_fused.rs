use cuda_device::{DisjointSlice, SharedArray, thread};

use super::cta_tile::{CTA_B_ELEMS, CtaMatmulDims, CtaTile};
use super::{CtaATile, CtaBTile};

const D: u32 = 64;
// An even, non-bank-multiple stride preserves the solve's diagonal access
// pattern while reducing its transposed stores to two-way bank conflicts.
const STRIDE: usize = 66;
pub(crate) const MATRIX_ELEMS: usize = D as usize * STRIDE;
pub(super) type InverseTile = SharedArray<f32, MATRIX_ELEMS>;

pub(super) fn inverse_body<const NATIVE_INPUTS: bool>(
    kpos: &[f32],
    kneg: *const u8,
    mut inverse: DisjointSlice<u16>,
    raw: &mut InverseTile,
    dims: CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_tile(dims.batch_count) else {
        return;
    };
    {
        let a = raw.as_mut_ptr() as *mut u16;
        let b = unsafe { a.add(super::cta_tile::CTA_A_ELEMS) };
        cta_accumulators!(acc0, acc1, acc2, acc3);
        let mut k_base = 0;
        while k_base < D {
            stage_inverse_arena::<NATIVE_INPUTS>(kpos, kneg, a, b, tile, k_base);
            thread::sync_threads();
            let af = arena_a_fragments(a, tile);
            crate::mma::mma_m16n8k16_f16_f16_f32(
                af,
                arena_b_fragments(b, tile, tile.warp_n0),
                &mut acc0,
            );
            crate::mma::mma_m16n8k16_f16_f16_f32(
                af,
                arena_b_fragments(b, tile, tile.warp_n0 + 1),
                &mut acc1,
            );
            crate::mma::mma_m16n8k16_f16_f16_f32(
                af,
                arena_b_fragments(b, tile, tile.warp_n0 + 2),
                &mut acc2,
            );
            crate::mma::mma_m16n8k16_f16_f16_f32(
                af,
                arena_b_fragments(b, tile, tile.warp_n0 + 3),
                &mut acc3,
            );
            super::cta_sync::sync_before_next_k(k_base, D);
            k_base += super::cta_tile::CTA_K;
        }
        // The arena changes roles only after all warps finish reading A/B.
        thread::sync_threads();
        cta_store4!(
            store_strict_lower_shared,
            tile,
            raw,
            dims,
            acc0,
            acc1,
            acc2,
            acc3
        );
    }
    thread::sync_threads();
    solve_strict_lower(tile.batch, &mut inverse, raw);
}

/// Invert I+L after a causal factor producer has formed strict-lower L.
/// This shares the established FP32 attention solve; it is not a projection or
/// an alternative dense model precision. The caller retains FP16 inverse tape.
pub(crate) fn formed_inverse_body(
    lower: &[f32],
    mut inverse: DisjointSlice<u16>,
    raw: &mut InverseTile,
    batch_count: u32,
) {
    let batch = thread::blockIdx_x();
    if batch >= batch_count {
        return;
    }
    let mut index = thread::threadIdx_x();
    while index < D * D {
        let row = index / D;
        let col = index % D;
        raw[row as usize * STRIDE + col as usize] = if col < row {
            lower[(batch * D * D + index) as usize]
        } else {
            0.0
        };
        index += thread::blockDim_x();
    }
    thread::sync_threads();
    solve_strict_lower(batch, &mut inverse, raw);
}

#[inline(always)]
fn solve_strict_lower(batch: u32, inverse: &mut DisjointSlice<u16>, raw: &mut InverseTile) {
    // Each column owns its transposed inverse row. The original strict-lower
    // product stays below the diagonal, exactly as in the separate FP32 solve.
    let col = thread::threadIdx_x();
    if col < D {
        let mut row = 0;
        while row < D {
            let value = if col > row {
                0.0
            } else if col == row {
                1.0
            } else {
                let mut sum = 0.0;
                let mut mid = col;
                while mid < row {
                    sum = crate::float_ptx::fma_f32(
                        raw[row as usize * STRIDE + mid as usize],
                        raw[col as usize * STRIDE + mid as usize],
                        sum,
                    );
                    mid += 1;
                }
                -sum
            };
            if col <= row {
                raw[col as usize * STRIDE + row as usize] = value;
            }
            let index = ((batch * D + row) * D + col) as usize;
            unsafe {
                *inverse.get_unchecked_mut(index) = super::convert::cvt_rn_f16_f32(value);
            }
            row += 1;
        }
    }
}

#[inline(always)]
fn stage_inverse_arena<const NATIVE_INPUTS: bool>(
    kpos: &[f32],
    kneg: *const u8,
    a: *mut u16,
    b: *mut u16,
    tile: CtaTile,
    k_base: u32,
) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < super::cta_tile::CTA_A_ELEMS as u32 {
        let (row, k) = super::cta_stage::stage_coords(pair, 0, k_base);
        let index = ((tile.batch * D + row) * D + k) as usize;
        let (a_bits, b_bits) = if NATIVE_INPUTS {
            (
                super::convert::load_f16x2_global_bits_read_only(
                    kpos.as_ptr() as *const u16,
                    crate::kda_common::kpos_half_arena_index(index),
                ),
                super::convert::load_f16x2_global_bits_read_only(kneg as *const u16, index),
            )
        } else {
            // Uncached backward reconstructs these products in FP32. This
            // compile-time specialization consumes that live reconstruction.
            let (a0, a1) = super::convert::load_f32x2_global_read_only(kpos.as_ptr(), index);
            let (b0, b1) = super::convert::load_f32x2_global_read_only(kneg as *const f32, index);
            (
                cuda_device::convert::cvt_f16x2_f32(a0, a1),
                cuda_device::convert::cvt_f16x2_f32(b0, b1),
            )
        };
        super::convert::store_f16x2_shared(a, pair as usize, a_bits);
        super::convert::store_f16x2_shared(b, pair as usize, b_bits);
        pair += thread::blockDim_x() * 2;
    }
}
#[inline(always)]
fn arena_packed(src: *const u16, row: u32, col: u32) -> u32 {
    unsafe { *(src.add((row * super::cta_tile::CTA_K + col) as usize) as *const u32) }
}
#[inline(always)]
fn arena_a_fragments(src: *const u16, tile: CtaTile) -> [u32; 4] {
    let row = tile.warp_m * 16 + tile.group;
    let col = tile.thread_in_group * 2;
    [
        arena_packed(src, row, col),
        arena_packed(src, row + 8, col),
        arena_packed(src, row, col + 8),
        arena_packed(src, row + 8, col + 8),
    ]
}
#[inline(always)]
fn arena_b_fragments(src: *const u16, tile: CtaTile, warp_n: u32) -> [u32; 2] {
    let row = warp_n * 8 + tile.group;
    let col = tile.thread_in_group * 2;
    [arena_packed(src, row, col), arena_packed(src, row, col + 8)]
}

#[inline(always)]
fn store_strict_lower_shared(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    raw: &mut InverseTile,
    _m: u32,
    _n: u32,
) {
    let mut i = 0;
    while i < 4 {
        let (row, col) = tile.accumulator_coords(warp_n, i);
        raw[row as usize * STRIDE + col as usize] = if col < row { acc[i] } else { 0.0 };
        i += 1;
    }
}

/// W = inverse @ Kpos, U = inverse @ Vbeta. Kpos's buffer becomes U.
pub(super) fn wu_body(
    inverse: &[u16],
    vbeta: &[u16],
    mut kpos_u: DisjointSlice<f32>,
    mut w: DisjointSlice<u16>,
    a_tile: &mut CtaATile,
    b_tiles: (&mut CtaBTile, &mut CtaBTile),
    dims: CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    let (b_w, b_u) = b_tiles;
    cta_accumulators!(w0, w1, u0, u1);
    let mut k_base = 0;
    while k_base < D {
        super::cta_stage::stage_matrix_tile::<false, { super::cta_tile::CTA_A_ELEMS }>(
            inverse, a_tile, tile, 0, D, D, k_base,
        );
        stage_inplace_rhs(kpos_u.as_mut_ptr(), b_w, tile, k_base);
        stage_native_rhs(vbeta.as_ptr(), b_u, tile, k_base);
        thread::sync_threads();
        cta_mma2!(a_tile, b_w, tile, w0, w1);
        cta_mma2!(a_tile, b_u, tile, u0, u1);
        super::cta_sync::sync_before_next_k(k_base, D);
        k_base += super::cta_tile::CTA_K;
    }
    cta_store2!(super::cta_store::store_half, tile, &mut w, dims, w0, w1);
    // One CTA owns each 64x64 chunk. The final staging barrier already finished
    // every coherent Kpos read before these same bytes become U.
    cta_store2!(super::cta_store::store, tile, &mut kpos_u, dims, u0, u1);
}

#[inline(always)]
fn stage_inplace_rhs(src: *const f32, b_tile: &mut CtaBTile, tile: CtaTile, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let (col, k) = super::cta_stage::stage_coords(pair, 0, k_base);
        let base = tile.batch * D * D;
        let lo = super::convert::load_f16_global_bits(
            src as *const u16,
            crate::kda_common::kpos_half_arena_index((base + k * D + col) as usize),
        );
        let hi = super::convert::load_f16_global_bits(
            src as *const u16,
            crate::kda_common::kpos_half_arena_index((base + (k + 1) * D + col) as usize),
        );
        super::convert::store_f16x2_shared(
            b_tile.as_mut_ptr(),
            pair as usize,
            u32::from(lo) | (u32::from(hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}

#[inline(always)]
fn stage_native_rhs(src: *const u16, b_tile: &mut CtaBTile, tile: CtaTile, k_base: u32) {
    let mut pair = thread::threadIdx_x() * 2;
    while pair < CTA_B_ELEMS as u32 {
        let (col, k) = super::cta_stage::stage_coords(pair, 0, k_base);
        let base = tile.batch * D * D;
        let lo = super::convert::load_f16_global_bits_read_only(src, (base + k * D + col) as usize);
        let hi = super::convert::load_f16_global_bits_read_only(
            src,
            (base + (k + 1) * D + col) as usize,
        );
        super::convert::store_f16x2_shared(
            b_tile.as_mut_ptr(),
            pair as usize,
            u32::from(lo) | (u32::from(hi) << 16),
        );
        pair += thread::blockDim_x() * 2;
    }
}
