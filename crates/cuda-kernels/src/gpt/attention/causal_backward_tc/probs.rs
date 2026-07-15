use cuda_device::{DisjointSlice, thread};

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::exp_f32;

pub(super) fn prob_ds_body(
    scores: &[f32],
    dot: &[f32],
    log_sum_exp: &[f32],
    softmax_d: &[f32],
    mut p: DisjointSlice<f32>,
    mut ds: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.seq_len;
    if index >= total {
        return;
    }

    let key = index % params.seq_len;
    let query = (index / params.seq_len) % params.seq_len;
    let batch_head = index / (params.seq_len * params.seq_len);
    let batch = batch_head / params.head_count;
    let head = batch_head - batch * params.head_count;
    let row = batch * params.seq_len + query;
    if key > query {
        return;
    }
    if row >= params.row_count {
        unsafe {
            *p.get_unchecked_mut(index as usize) = 0.0;
            *ds.get_unchecked_mut(index as usize) = 0.0;
        }
        return;
    }

    let lse_index = log_sum_exp_index(batch, query, head, &params);
    let prob = exp_f32(scores[index as usize] * params.scale - log_sum_exp[lse_index]);
    let grad = prob * (dot[index as usize] - softmax_d[lse_index]);

    unsafe {
        *p.get_unchecked_mut(index as usize) = prob;
        *ds.get_unchecked_mut(index as usize) = grad;
    }
}

pub(super) fn prob_ds_f16_body(
    scores: &[f32],
    dot: &[f32],
    log_sum_exp: &[f32],
    softmax_d: &[f32],
    mut p: DisjointSlice<u16>,
    mut ds: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.seq_len;
    if index >= total {
        return;
    }

    let key = index % params.seq_len;
    let query = (index / params.seq_len) % params.seq_len;
    let batch_head = index / (params.seq_len * params.seq_len);
    let batch = batch_head / params.head_count;
    let head = batch_head - batch * params.head_count;
    let row = batch * params.seq_len + query;
    if key > query {
        return;
    }
    if row >= params.row_count {
        unsafe {
            *p.get_unchecked_mut(index as usize) = 0;
            *ds.get_unchecked_mut(index as usize) = 0;
        }
        return;
    }

    let lse_index = log_sum_exp_index(batch, query, head, &params);
    let prob = exp_f32(scores[index as usize] * params.scale - log_sum_exp[lse_index]);
    let grad = prob * (dot[index as usize] - softmax_d[lse_index]);

    unsafe {
        *p.get_unchecked_mut(index as usize) = cvt_rn_f16_f32(prob);
        *ds.get_unchecked_mut(index as usize) = cvt_rn_f16_f32(grad);
    }
}

#[inline(always)]
fn log_sum_exp_index(batch: u32, token: u32, head: u32, params: &CausalAttentionParams) -> usize {
    (batch as usize * params.head_count as usize + head as usize) * params.seq_len as usize
        + token as usize
}
