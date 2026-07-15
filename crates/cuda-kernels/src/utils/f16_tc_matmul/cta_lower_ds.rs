use cuda_device::{DisjointSlice, thread};

use super::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use super::cta_tile::{CtaMatmulDims, CtaTile};

pub(super) fn cta_matmul_lower_ds_body(
    a: &[u16],
    b_t: &[u16],
    probs: &[u16],
    softmax_d: &[f32],
    mut out: DisjointSlice<u16>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: CtaMatmulDims,
) {
    if thread::blockIdx_x() > thread::blockIdx_y() {
        return;
    }
    let Some(tile) = super::cta_tile::active_tile(dims.batch_count) else {
        return;
    };
    let aligned = dims.aligned();
    cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
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
    store_ds(
        acc2,
        tile,
        tile.warp_n0 + 2,
        probs,
        softmax_d,
        &mut out,
        dims,
    );
    store_ds(
        acc3,
        tile,
        tile.warp_n0 + 3,
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
    store_one(acc[0], 0, tile, warp_n, probs, softmax_d, out, dims);
    store_one(acc[1], 1, tile, warp_n, probs, softmax_d, out, dims);
    store_one(acc[2], 2, tile, warp_n, probs, softmax_d, out, dims);
    store_one(acc[3], 3, tile, warp_n, probs, softmax_d, out, dims);
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
        let d = softmax_d[(tile.batch * dims.m + row) as usize];
        let grad = cvt_f32_f16(probs[offset]) * (dot - d);
        unsafe {
            *out.get_unchecked_mut(offset) = cvt_rn_f16_f32(grad);
        }
    }
}
