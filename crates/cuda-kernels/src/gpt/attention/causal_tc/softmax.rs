use cuda_device::{DisjointSlice, SharedArray, thread};

use super::gather::TC_FORWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::{exp_f32, ln_f32, max_f32, safe_positive_denom};

mod index;
mod reduce;

use index::{log_sum_exp_index, score, score_index};
use reduce::{block_reduce_max, block_reduce_sum};

pub(super) const WARPS_PER_BLOCK: usize = (TC_FORWARD_THREADS_PER_BLOCK / 32) as usize;
pub(super) const NEG_INFINITY: f32 = -3.4028235e38_f32;

#[derive(Clone, Copy)]
struct SoftmaxRow<'a> {
    batch: u32,
    head: u32,
    query: u32,
    tid: u32,
    params: &'a CausalAttentionParams,
}

pub(super) fn softmax_body(
    scores: &[f32],
    mut probs: DisjointSlice<f32>,
    mut log_sum_exp: DisjointSlice<f32>,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) {
    let query = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let batch = thread::blockIdx_z();
    let tid = thread::threadIdx_x();
    let row = batch * params.seq_len + query;
    let ctx = SoftmaxRow {
        batch,
        head,
        query,
        tid,
        params: &params,
    };
    if row >= params.row_count {
        zero_prob_row(&mut probs, ctx);
        if tid == 0 {
            unsafe {
                *log_sum_exp.get_unchecked_mut(log_sum_exp_index(batch, query, head, ctx.params)) =
                    0.0;
            }
        }
        return;
    }

    let max_score = query_max(scores, ctx, reduce);
    let denom = safe_positive_denom(query_denom(scores, ctx, max_score, reduce));
    if tid == 0 {
        unsafe {
            *log_sum_exp.get_unchecked_mut(log_sum_exp_index(batch, query, head, ctx.params)) =
                max_score + ln_f32(denom);
        }
    }

    let mut key = tid;
    while key < params.seq_len {
        let prob = if params.key_is_visible(query, key) {
            exp_f32(score(scores, batch, head, query, key, ctx.params) - max_score) / denom
        } else {
            0.0
        };
        unsafe {
            *probs.get_unchecked_mut(score_index(batch, head, query, key, ctx.params)) = prob;
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

pub(super) fn softmax_f16_body(
    scores: &[f32],
    mut probs: DisjointSlice<u16>,
    mut log_sum_exp: DisjointSlice<f32>,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) {
    let query = thread::blockIdx_x();
    let head = thread::blockIdx_y();
    let batch = thread::blockIdx_z();
    let tid = thread::threadIdx_x();
    let row = batch * params.seq_len + query;
    let ctx = SoftmaxRow {
        batch,
        head,
        query,
        tid,
        params: &params,
    };
    if row >= params.row_count {
        zero_prob_row_f16(&mut probs, ctx);
        if tid == 0 {
            unsafe {
                *log_sum_exp.get_unchecked_mut(log_sum_exp_index(batch, query, head, ctx.params)) =
                    0.0;
            }
        }
        return;
    }

    let max_score = query_max(scores, ctx, reduce);
    let denom = safe_positive_denom(query_denom(scores, ctx, max_score, reduce));
    if tid == 0 {
        unsafe {
            *log_sum_exp.get_unchecked_mut(log_sum_exp_index(batch, query, head, ctx.params)) =
                max_score + ln_f32(denom);
        }
    }

    let first_key = params.first_visible_key(query);
    let mut key = tid;
    while key < first_key {
        unsafe {
            *probs.get_unchecked_mut(score_index(batch, head, query, key, ctx.params)) = 0;
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }

    let mut key = first_key + tid;
    while key <= query {
        let prob = exp_f32(score(scores, batch, head, query, key, ctx.params) - max_score) / denom;
        unsafe {
            *probs.get_unchecked_mut(score_index(batch, head, query, key, ctx.params)) =
                cvt_rn_f16_f32(prob);
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn zero_prob_row(probs: &mut DisjointSlice<f32>, ctx: SoftmaxRow<'_>) {
    let mut key = ctx.tid;
    while key < ctx.params.seq_len {
        unsafe {
            *probs
                .get_unchecked_mut(score_index(ctx.batch, ctx.head, ctx.query, key, ctx.params)) =
                0.0;
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn zero_prob_row_f16(probs: &mut DisjointSlice<u16>, ctx: SoftmaxRow<'_>) {
    let mut key = ctx.tid;
    while key < ctx.params.seq_len {
        unsafe {
            *probs
                .get_unchecked_mut(score_index(ctx.batch, ctx.head, ctx.query, key, ctx.params)) =
                0;
        }
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
}

fn query_max(
    scores: &[f32],
    ctx: SoftmaxRow<'_>,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let mut local = NEG_INFINITY;
    let mut key = ctx.params.first_visible_key(ctx.query) + ctx.tid;
    while key <= ctx.query {
        local = max_f32(
            local,
            score(scores, ctx.batch, ctx.head, ctx.query, key, ctx.params),
        );
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
    block_reduce_max(local, ctx.tid, reduce)
}

fn query_denom(
    scores: &[f32],
    ctx: SoftmaxRow<'_>,
    max_score: f32,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let mut local = 0.0;
    let mut key = ctx.params.first_visible_key(ctx.query) + ctx.tid;
    while key <= ctx.query {
        local +=
            exp_f32(score(scores, ctx.batch, ctx.head, ctx.query, key, ctx.params) - max_score);
        key += TC_FORWARD_THREADS_PER_BLOCK;
    }
    block_reduce_sum(local, ctx.tid, reduce)
}
