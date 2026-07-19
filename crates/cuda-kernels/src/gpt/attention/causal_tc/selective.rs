use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;

pub(super) const SELECTIVE_ATTENTION_THREADS_PER_BLOCK: u32 = 32;
pub(super) const SELECTIVE_APPLY_THREADS_PER_BLOCK: u32 = 256;

pub(super) fn selective_attention_mask_body(
    scores: &[f32],
    mask_values: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    selective_attention_mask_impl::<false>(scores, mask_values, core::ptr::null_mut(), params);
}

pub(super) fn selective_attention_mask_save_tape_body(
    scores: &[f32],
    mask_values: DisjointSlice<f32>,
    mut selection_mask_tape: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    selective_attention_mask_impl::<true>(
        scores,
        mask_values,
        selection_mask_tape.as_mut_ptr(),
        params,
    );
}

fn selective_attention_mask_impl<const SAVE_MASK: bool>(
    scores: &[f32],
    mut mask_values: DisjointSlice<f32>,
    selection_mask_tape: *mut u16,
    params: CausalAttentionParams,
) {
    let key_tiles = params
        .seq_len
        .div_ceil(SELECTIVE_ATTENTION_THREADS_PER_BLOCK);
    let block = thread::blockIdx_x();
    let batch = block / key_tiles;
    if batch >= params.batch_size {
        return;
    }

    let key_tile = block - batch * key_tiles;
    let key_base = key_tile * SELECTIVE_ATTENTION_THREADS_PER_BLOCK;
    let key = key_base + thread::threadIdx_x();
    let last_key_unclamped = key_base + SELECTIVE_ATTENTION_THREADS_PER_BLOCK - 1;
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
    let mut cumulative_mask = 0.0_f32;
    let mut query = key_base;
    while query <= last_query {
        let row = batch * params.seq_len + query;
        let visible = key < params.seq_len
            && query >= key
            && query - key < params.attention_window
            && row < params.row_count;
        if visible {
            let selection_index = score_index(batch, 0, query, key, &params);
            let raw_selection_score = scores[selection_index];
            let distance = query - key;
            let mask_index = packed_mask_index(batch, query, distance, &params);
            unsafe {
                *mask_values.get_unchecked_mut(mask_index) = cumulative_mask;
            }

            let can_select = key != 0 && query != key;
            if SAVE_MASK && can_select {
                unsafe {
                    *selection_mask_tape.add(tape_base + mask_index) =
                        u16::from(raw_selection_score > 0.0);
                }
            }
            if can_select && raw_selection_score > 0.0 {
                cumulative_mask += raw_selection_score;
            }
        }
        query += 1;
    }
}

pub(super) fn apply_selective_attention_mask_body(
    mut scores: DisjointSlice<f32>,
    mask_values: &[f32],
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * SELECTIVE_APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
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
    let mask = mask_values[index as usize];
    let mut head = 0;
    while head < params.head_count {
        let index = score_index(batch, head, query, key, &params);
        unsafe {
            let score = *scores.as_mut_ptr().add(index);
            *scores.get_unchecked_mut(index) = score - mask;
        }
        head += 1;
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
