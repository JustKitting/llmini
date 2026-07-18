use cuda_device::{DisjointSlice, convert::cvt_f16x2_f32, thread};

use super::convert::{
    cvt_f32_f16, cvt_rn_f16_f32, load_f16_global_bits_read_only, load_f16x2_global_bits_read_only,
    load_f32_global_read_only, store_f16x2_global,
};
use super::cta_tile::{CtaMatmulDims, CtaTile};

pub(super) fn cta_matmul_lower_ds_body(
    a: &[u16],
    b_t: &[u16],
    probs: &[u16],
    softmax_d: &[f32],
    out: DisjointSlice<u16>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: CtaMatmulDims,
) {
    cta_matmul_windowed_lower_ds_body(a, b_t, probs, softmax_d, out, a_tile, b_tile, dims, dims.m);
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn cta_matmul_windowed_lower_ds_body(
    a: &[u16],
    b_t: &[u16],
    probs: &[u16],
    softmax_d: &[f32],
    mut out: DisjointSlice<u16>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: CtaMatmulDims,
    window: u32,
) {
    let tile_col = thread::blockIdx_x();
    let tile_row = thread::blockIdx_y();
    let window_tiles = window / super::cta_tile::CTA_N;
    if tile_col > tile_row || tile_row > tile_col + window_tiles {
        return;
    }
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    let aligned = dims.aligned();
    cta_accumulate_k_loop2!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1]; {
        if aligned {
            super::cta_stage::stage_tiles_aligned(a, b_t, a_tile, b_tile, tile, dims, k_base);
        } else {
            super::cta_stage::stage_tiles(a, b_t, a_tile, b_tile, tile, dims, k_base);
        }
    });
    store_ds(acc0, tile, tile.warp_n0, probs, softmax_d, &mut out, dims);
    store_ds(
        acc1,
        tile,
        tile.warp_n0 + 1,
        probs,
        softmax_d,
        &mut out,
        dims,
    );
}

#[inline(always)]
fn store_ds(
    acc: [f32; 4],
    tile: CtaTile,
    warp_n: u32,
    probs: &[u16],
    softmax_d: &[f32],
    out: &mut DisjointSlice<u16>,
    dims: CtaMatmulDims,
) {
    store_pair(acc[0], acc[1], 0, tile, warp_n, probs, softmax_d, out, dims);
    store_pair(acc[2], acc[3], 2, tile, warp_n, probs, softmax_d, out, dims);
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA store coordinates are explicit"
)]
fn store_pair(
    dot0: f32,
    dot1: f32,
    acc_index: usize,
    tile: CtaTile,
    warp_n: u32,
    probs: &[u16],
    softmax_d: &[f32],
    out: &mut DisjointSlice<u16>,
    dims: CtaMatmulDims,
) {
    let (row0, col0) = tile.accumulator_coords(warp_n, acc_index);
    let (row1, col1) = tile.accumulator_coords(warp_n, acc_index + 1);
    if row0 < dims.m && row1 == row0 && col0 + 1 == col1 && col1 < dims.n && col1 <= row0 {
        let offset = ((tile.batch * dims.m + row0) * dims.n + col0) as usize;
        let packed_probs = load_f16x2_global_bits_read_only(probs.as_ptr(), offset);
        let d =
            load_f32_global_read_only(softmax_d.as_ptr(), (tile.batch * dims.m + row0) as usize);
        let grad0 = cvt_f32_f16(packed_probs as u16) * (dot0 - d);
        let grad1 = cvt_f32_f16((packed_probs >> 16) as u16) * (dot1 - d);
        store_f16x2_global(out.as_mut_ptr(), offset, cvt_f16x2_f32(grad0, grad1));
    } else {
        store_one(dot0, acc_index, tile, warp_n, probs, softmax_d, out, dims);
        store_one(
            dot1,
            acc_index + 1,
            tile,
            warp_n,
            probs,
            softmax_d,
            out,
            dims,
        );
    }
}

#[inline(always)]
fn store_one(
    dot: f32,
    acc_index: usize,
    tile: CtaTile,
    warp_n: u32,
    probs: &[u16],
    softmax_d: &[f32],
    out: &mut DisjointSlice<u16>,
    dims: CtaMatmulDims,
) {
    let (row, col) = tile.accumulator_coords(warp_n, acc_index);
    if row < dims.m && col < dims.n && col <= row {
        let offset = ((tile.batch * dims.m + row) * dims.n + col) as usize;
        let d = load_f32_global_read_only(softmax_d.as_ptr(), (tile.batch * dims.m + row) as usize);
        let grad = cvt_f32_f16(load_f16_global_bits_read_only(probs.as_ptr(), offset)) * (dot - d);
        unsafe {
            *out.get_unchecked_mut(offset) = cvt_rn_f16_f32(grad);
        }
    }
}
