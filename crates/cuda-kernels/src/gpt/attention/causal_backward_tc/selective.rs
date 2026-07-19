use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::f16_tc_matmul::cta_tile::{CTA_M, CTA_N};

pub(super) const SELECTIVE_BACKWARD_THREADS_PER_BLOCK: u32 = 32;
pub(super) const SELECTIVE_ROW_SUM_THREADS_PER_BLOCK: u32 = 256;

pub(super) fn selective_attention_row_sums_dense_body(
    ds: &[u16],
    row_sums: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    selective_attention_row_sums_impl::<false>(ds, &[], row_sums, params);
}

pub(super) fn selective_attention_row_sums_sparse_body(
    ds: &[u16],
    tile_scales: &[f32],
    row_sums: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    selective_attention_row_sums_impl::<true>(ds, tile_scales, row_sums, params);
}

fn selective_attention_row_sums_impl<const SPARSE: bool>(
    ds: &[u16],
    tile_scales: &[f32],
    mut row_sums: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * SELECTIVE_ROW_SUM_THREADS_PER_BLOCK + thread::threadIdx_x();
    let packed_per_batch = params.seq_len * params.attention_window;
    let total = params.batch_size * packed_per_batch;
    if index >= total {
        return;
    }

    let batch = index / packed_per_batch;
    let packed_in_batch = index - batch * packed_per_batch;
    let query = packed_in_batch / params.attention_window;
    let distance = packed_in_batch - query * params.attention_window;
    if distance > query {
        return;
    }
    let row = batch * params.seq_len + query;
    if row >= params.row_count {
        return;
    }

    let key = query - distance;
    let mut row_logit_grad = 0.0_f32;
    let mut head = 0;
    while head < params.head_count {
        let score_index = score_index(batch, head, query, key, &params);
        let active = !SPARSE || tile_scale(tile_scales, batch, head, query, key, &params) != 0.0;
        if active {
            row_logit_grad += cvt_f32_f16(ds[score_index]);
        }
        head += 1;
    }
    unsafe {
        *row_sums.get_unchecked_mut(index as usize) = row_logit_grad;
    }
}

pub(super) fn selective_attention_backward_dense_body(
    selection_mask_tape: &[u16],
    row_sums: &[f32],
    ds: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    selective_attention_backward_impl::<false>(selection_mask_tape, &[], row_sums, ds, params);
}

pub(super) fn selective_attention_backward_sparse_body(
    selection_mask_tape: &[u16],
    tile_scales: &[f32],
    row_sums: &[f32],
    ds: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    selective_attention_backward_impl::<true>(
        selection_mask_tape,
        tile_scales,
        row_sums,
        ds,
        params,
    );
}

fn selective_attention_backward_impl<const SPARSE: bool>(
    selection_mask_tape: &[u16],
    tile_scales: &[f32],
    row_sums: &[f32],
    mut ds: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let key_tiles = params
        .seq_len
        .div_ceil(SELECTIVE_BACKWARD_THREADS_PER_BLOCK);
    let block = thread::blockIdx_x();
    let batch = block / key_tiles;
    if batch >= params.batch_size {
        return;
    }

    let key_tile = block - batch * key_tiles;
    let key_base = key_tile * SELECTIVE_BACKWARD_THREADS_PER_BLOCK;
    let key = key_base + thread::threadIdx_x();
    let last_key_unclamped = key_base + SELECTIVE_BACKWARD_THREADS_PER_BLOCK - 1;
    let last_key = if last_key_unclamped < params.seq_len {
        last_key_unclamped
    } else {
        params.seq_len - 1
    };
    let last_window_query = last_key + params.attention_window - 1;
    let last_query = if last_window_query < params.seq_len {
        last_window_query
    } else {
        params.seq_len - 1
    };
    let tape_base = params.row_count as usize * params.qkv_dim as usize;
    let mut future_logit_grad = 0.0_f32;
    let mut query = last_query + 1;
    while query > key_base {
        query -= 1;
        let row = batch * params.seq_len + query;
        let visible = key < params.seq_len
            && query >= key
            && query - key < params.attention_window
            && row < params.row_count;
        if visible {
            let distance = query - key;
            let packed_index = packed_mask_index(batch, query, distance, &params);
            let head_zero_index = score_index(batch, 0, query, key, &params);
            let direct_active =
                !SPARSE || tile_scale(tile_scales, batch, 0, query, key, &params) != 0.0;
            let direct_selection_head_grad = if direct_active {
                cvt_f32_f16(unsafe { *ds.as_mut_ptr().add(head_zero_index) })
            } else {
                0.0
            };
            let selection_grad =
                if key != 0 && query != key && selection_mask_tape[tape_base + packed_index] != 0 {
                    -future_logit_grad
                } else {
                    0.0
                };
            unsafe {
                *ds.get_unchecked_mut(head_zero_index) =
                    cvt_rn_f16_f32(direct_selection_head_grad + selection_grad);
            }
            future_logit_grad += row_sums[packed_index];
        }
    }
}

#[inline(always)]
fn packed_mask_index(
    batch: u32,
    query: u32,
    distance: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((batch * params.seq_len + query) * params.attention_window + distance) as usize
}

pub(super) fn activate_selective_head_tiles_body(
    mut tile_scales: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * SELECTIVE_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let tiles = params.seq_len.div_ceil(CTA_M);
    let tiles_per_batch = tiles * tiles;
    let total = params.batch_size * tiles_per_batch;
    if index >= total {
        return;
    }

    let batch = index / tiles_per_batch;
    let tile_in_batch = index - batch * tiles_per_batch;
    let query_tile = tile_in_batch / tiles;
    let key_tile = tile_in_batch - query_tile * tiles;
    let window_tiles = params.attention_window / CTA_N;
    if key_tile > query_tile || query_tile > key_tile + window_tiles {
        return;
    }
    let batch_head = batch * params.head_count;
    let tile_index = (batch_head * tiles + query_tile) * tiles + key_tile;
    unsafe {
        *tile_scales.get_unchecked_mut(tile_index as usize) = 1.0;
    }
}

#[inline(always)]
fn tile_scale(
    tile_scales: &[f32],
    batch: u32,
    head: u32,
    query: u32,
    key: u32,
    params: &CausalAttentionParams,
) -> f32 {
    let tiles = params.seq_len.div_ceil(CTA_M);
    let query_tile = query / CTA_M;
    let key_tile = key / CTA_N;
    let batch_head = batch * params.head_count + head;
    tile_scales[((batch_head * tiles + query_tile) * tiles + key_tile) as usize]
}

#[inline(always)]
fn score_index(
    batch: u32,
    head: u32,
    query: u32,
    key: u32,
    params: &CausalAttentionParams,
) -> usize {
    (((batch as usize * params.head_count as usize + head as usize) * params.seq_len as usize
        + query as usize)
        * params.seq_len as usize)
        + key as usize
}
