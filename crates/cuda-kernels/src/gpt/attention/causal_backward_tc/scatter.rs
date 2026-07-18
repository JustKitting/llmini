use cuda_device::{DisjointSlice, SharedArray, thread};

use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::attention::layout::{
    batched_qkv_index, compact_index, compact_linear_parts, qkv_value, row_index,
};
use crate::block_reduce::block_sum_shared_f32;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{abs_f32, exp_f32, fma_f32, max_f32, sincos_f32};
use crate::nvfp4::nvfp4_value;
use crate::warp_reduce::warp_sum_f32;

pub(super) fn scatter_body(
    d_q: &[f32],
    d_k: &[f32],
    d_v: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let index = thread::blockIdx_x() * TC_BACKWARD_THREADS_PER_BLOCK + thread::threadIdx_x();
    let total = params.batch_size * params.head_count * params.seq_len * params.head_dim;
    if index >= total {
        return;
    }

    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let row = row_index(batch, token, &params);
    let q = batched_qkv_index(batch, token, head, dim, 0, &params);
    let k = batched_qkv_index(batch, token, head, dim, params.embedding_dim, &params);
    let v = batched_qkv_index(batch, token, head, dim, params.embedding_dim * 2, &params);
    if row >= params.row_count {
        unsafe {
            *d_qkv.get_unchecked_mut(q) = 0.0;
            *d_qkv.get_unchecked_mut(k) = 0.0;
            *d_qkv.get_unchecked_mut(v) = 0.0;
        }
        return;
    }

    let pair_index = index ^ 1;
    let dq = rope_raw_grad(
        token,
        dim,
        d_q[index as usize] * params.scale,
        d_q[pair_index as usize] * params.scale,
        params.head_dim,
    );
    let dk = rope_raw_grad(
        token,
        dim,
        d_k[index as usize] * params.scale,
        d_k[pair_index as usize] * params.scale,
        params.head_dim,
    );

    unsafe {
        *d_qkv.get_unchecked_mut(q) = dq;
        *d_qkv.get_unchecked_mut(k) = dk;
        *d_qkv.get_unchecked_mut(v) = d_v[index as usize];
    }
}

pub(super) fn scatter_amax_body(
    d_q: &[f32],
    d_k: &[f32],
    d_v: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    params: CausalAttentionParams,
) -> f32 {
    let chunk = thread::blockIdx_x();
    let row = chunk / 3;
    let section = chunk - row * 3;
    let batch = row / params.seq_len;
    let token = row - batch * params.seq_len;
    let mut col = thread::threadIdx_x();
    let mut local_amax = 0.0;

    while col < params.embedding_dim {
        let head = col / params.head_dim;
        let dim = col - head * params.head_dim;
        let compact = compact_index(batch, token, head, dim, &params);
        let value = if section == 0 {
            rope_raw_grad(
                token,
                dim,
                d_q[compact] * params.scale,
                d_q[compact ^ 1] * params.scale,
                params.head_dim,
            )
        } else if section == 1 {
            rope_raw_grad(
                token,
                dim,
                d_k[compact] * params.scale,
                d_k[compact ^ 1] * params.scale,
                params.head_dim,
            )
        } else {
            d_v[compact]
        };
        let out = row as usize * params.qkv_dim as usize
            + section as usize * params.embedding_dim as usize
            + col as usize;
        unsafe {
            *d_qkv.get_unchecked_mut(out) = value;
        }
        local_amax = max_f32(local_amax, abs_f32(value));
        col += TC_BACKWARD_THREADS_PER_BLOCK;
    }

    local_amax
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA scatter exposes explicit buffers"
)]
pub(super) fn scatter_qknorm_amax_body(
    qkv: &[u16],
    q_norms: &[f32],
    k_norms: &[f32],
    d_q: &[f32],
    d_k: &[f32],
    d_v: &[f32],
    qk_scale_bytes: &[u8],
    qk_scale_scales: &[u8],
    qk_scale_global_scale: &[f32],
    mut d_qkv: DisjointSlice<f32>,
    mut qk_scale_rows: DisjointSlice<f32>,
    params: CausalAttentionParams,
    qk_dot_warp_sums: &mut SharedArray<f32, 8>,
    scale_grad_warp_sums: &mut SharedArray<f32, 8>,
) -> f32 {
    let chunk = thread::blockIdx_x();
    let row = chunk / 3;
    let section = chunk - row * 3;
    let batch = row / params.seq_len;
    let token = row - batch * params.seq_len;
    let tid = thread::threadIdx_x();
    let lane = tid & 31;
    let warp = tid / 32;
    let learned_scale = nvfp4_value(qk_scale_bytes, qk_scale_scales, qk_scale_global_scale[0], 0);
    let mut local_amax = 0.0;
    let mut local_scale_grad = 0.0;
    let column_tiles =
        (params.embedding_dim + TC_BACKWARD_THREADS_PER_BLOCK - 1) / TC_BACKWARD_THREADS_PER_BLOCK;
    let mut column_tile = 0;

    while column_tile < column_tiles {
        let col = column_tile * TC_BACKWARD_THREADS_PER_BLOCK + tid;
        let valid_col = col < params.embedding_dim;
        if section < 2 {
            let section_offset = section * params.embedding_dim;
            let mut head = 0;
            let mut dim = 0;
            let mut compact = 0;
            let mut norm = 1.0;
            let mut raw = 0.0;
            let mut grad = 0.0;
            if valid_col {
                head = col / params.head_dim;
                dim = col - head * params.head_dim;
                compact = compact_index(batch, token, head, dim, &params);
                let norm_index = (head * params.row_count + row) as usize;
                norm = if section == 0 {
                    q_norms[norm_index]
                } else {
                    k_norms[norm_index]
                };
                raw = cvt_f32_f16(qkv_value(
                    qkv,
                    batch,
                    token,
                    head,
                    dim,
                    section_offset,
                    &params,
                ));
                grad = if section == 0 {
                    d_q[compact]
                } else {
                    d_k[compact]
                };
            }
            let normalized = if valid_col { raw / norm } else { 0.0 };
            let dot_contribution = grad * normalized;
            let warp_sum = warp_sum_f32(dot_contribution);
            if lane == 0 {
                qk_dot_warp_sums[warp as usize] = warp_sum;
            }
            thread::sync_threads();

            if valid_col {
                let first_warp = (tid / params.head_dim) * (params.head_dim / 32);
                let head_dot = qk_dot_warp_sums[first_warp as usize]
                    + qk_dot_warp_sums[first_warp as usize + 1];
                let jacobian_scale = if section == 0 {
                    learned_scale / norm
                } else {
                    params.scale / norm
                };
                let grad_dim = (grad - normalized * head_dot) * jacobian_scale;

                let pair_dim = dim ^ 1;
                let pair_compact = compact ^ 1;
                let pair_raw = cvt_f32_f16(qkv_value(
                    qkv,
                    batch,
                    token,
                    head,
                    pair_dim,
                    section_offset,
                    &params,
                ));
                let pair_grad = if section == 0 {
                    d_q[pair_compact]
                } else {
                    d_k[pair_compact]
                };
                let pair_normalized = pair_raw / norm;
                let grad_pair = (pair_grad - pair_normalized * head_dot) * jacobian_scale;
                let value = rope_raw_grad(token, dim, grad_dim, grad_pair, params.head_dim);
                let out = row as usize * params.qkv_dim as usize
                    + section as usize * params.embedding_dim as usize
                    + col as usize;
                unsafe {
                    *d_qkv.get_unchecked_mut(out) = value;
                }
                local_amax = max_f32(local_amax, abs_f32(value));
                if section == 0 {
                    local_scale_grad += dot_contribution;
                }
            }
            // Every warp must finish consuming this tile's shared reductions
            // before any warp overwrites them for the next group of heads.
            thread::sync_threads();
        } else if valid_col {
            let head = col / params.head_dim;
            let dim = col - head * params.head_dim;
            let compact = compact_index(batch, token, head, dim, &params);
            let value = d_v[compact];
            let out = row as usize * params.qkv_dim as usize
                + section as usize * params.embedding_dim as usize
                + col as usize;
            unsafe {
                *d_qkv.get_unchecked_mut(out) = value;
            }
            local_amax = max_f32(local_amax, abs_f32(value));
        }
        column_tile += 1;
    }

    if section == 0 {
        let row_scale_grad =
            block_sum_shared_f32(scale_grad_warp_sums, local_scale_grad, lane, warp);
        if tid == 0 {
            unsafe {
                *qk_scale_rows.get_unchecked_mut(row as usize) = row_scale_grad;
            }
        }
    }

    local_amax
}

#[inline(always)]
fn rope_raw_grad(token: u32, dim: u32, grad_dim: f32, grad_pair: f32, head_dim: u32) -> f32 {
    let (sin, cos) = sincos_f32(token as f32 * rope_inv_freq(dim, head_dim));
    if dim & 1 == 0 {
        fma_f32(grad_pair, sin, grad_dim * cos)
    } else {
        fma_f32(-grad_pair, sin, grad_dim * cos)
    }
}

#[inline(always)]
fn rope_inv_freq(dim: u32, head_dim: u32) -> f32 {
    exp_f32(-9.210_340_5 * (dim & !1) as f32 / head_dim as f32)
}
