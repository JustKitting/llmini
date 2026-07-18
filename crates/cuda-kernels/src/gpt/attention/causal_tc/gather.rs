use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::layout::{batched_qkv_index, compact_linear_parts, row_index};
use crate::attention::{CausalAttentionParams, QK_NORM_EPS};
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::float_ptx::{max_f32, sqrt_f32};
use crate::nvfp4::nvfp4_value;
use crate::warp_reduce::warp_sum_f32;

pub(super) const TC_FORWARD_THREADS_PER_BLOCK: u32 = 256;

pub(super) fn gather_qkv_body(
    qkv: &[f32],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    if row >= params.row_count {
        unsafe {
            *q.get_unchecked_mut(index as usize) = 0.0;
            *k.get_unchecked_mut(index as usize) = 0.0;
            *v.get_unchecked_mut(index as usize) = 0.0;
        }
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, 0, &params)];
        *k.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params)];
        *v.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params)];
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA gather exposes explicit buffers"
)]
pub(super) fn gather_qknorm_v_f16_body(
    qkv: &[f32],
    qk_scale_bytes: &[u8],
    qk_scale_scales: &[u8],
    qk_scale_global_scale: &[f32],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<u16>,
    params: CausalAttentionParams,
    q_warp_sums: &mut SharedArray<f32, 8>,
    k_warp_sums: &mut SharedArray<f32, 8>,
) {
    let tid = thread::threadIdx_x();
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + tid;
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    let mut q_value = 0.0;
    let mut k_value = 0.0;
    let mut v_value = 0;
    let mut valid_row = false;

    if index < total {
        let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
        let row = row_index(batch, token, &params);
        valid_row = row < params.row_count;
        if valid_row {
            q_value = qkv[batched_qkv_index(batch, token, head, dim, 0, &params)];
            k_value =
                qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params)];
            v_value = cvt_rn_f16_f32(
                qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params)],
            );
        }
    }

    let lane = tid & 31;
    let warp = tid / 32;
    let q_warp_sum = warp_sum_f32(q_value * q_value);
    let k_warp_sum = warp_sum_f32(k_value * k_value);
    if lane == 0 {
        q_warp_sums[warp as usize] = q_warp_sum;
        k_warp_sums[warp as usize] = k_warp_sum;
    }
    thread::sync_threads();

    if index < total {
        let head_group = tid / params.head_dim;
        let first_warp = head_group * (params.head_dim / 32);
        let q_norm = max_f32(
            sqrt_f32(q_warp_sums[first_warp as usize] + q_warp_sums[first_warp as usize + 1]),
            QK_NORM_EPS,
        );
        let k_norm = max_f32(
            sqrt_f32(k_warp_sums[first_warp as usize] + k_warp_sums[first_warp as usize + 1]),
            QK_NORM_EPS,
        );
        let learned_scale =
            nvfp4_value(qk_scale_bytes, qk_scale_scales, qk_scale_global_scale[0], 0);
        let q_scale = learned_scale / params.scale;
        unsafe {
            *q.get_unchecked_mut(index as usize) = if valid_row {
                q_value * (q_scale / q_norm)
            } else {
                0.0
            };
            *k.get_unchecked_mut(index as usize) = if valid_row { k_value / k_norm } else { 0.0 };
            *v.get_unchecked_mut(index as usize) = if valid_row { v_value } else { 0 };
        }
    }
}

pub(super) fn gather_qk_v_f16_body(
    qkv: &[f32],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_FORWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    if row >= params.row_count {
        unsafe {
            *q.get_unchecked_mut(index as usize) = 0.0;
            *k.get_unchecked_mut(index as usize) = 0.0;
            *v.get_unchecked_mut(index as usize) = 0;
        }
        return;
    }

    unsafe {
        *q.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, 0, &params)];
        *k.get_unchecked_mut(index as usize) =
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params)];
        *v.get_unchecked_mut(index as usize) = cvt_rn_f16_f32(
            qkv[batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params)],
        );
    }
}
