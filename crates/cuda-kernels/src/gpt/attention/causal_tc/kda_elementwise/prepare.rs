use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::thread_index;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::kda_common::{
    batch_head, beta_index, beta_offset, chunk_g_last_index, compact_index, g_offset, k_offset,
    kda_decay_exp, q_offset, qkv_index, safe_denom, sigmoid, softplus, v_offset, value_activation,
};
use crate::kda_elementwise::{KdaQkAct, KdaQkNormAcc, read_qk_act};

pub(in super::super) const KDA_FUSED_PREPARE_TILE_ELEMS: usize = 64 * 64;
pub(in super::super) const KDA_FUSED_PREPARE_BETA_ELEMS: usize = 64;

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn prepare_transform_kda_body(
    qkv: &[f32],
    qkv_f16: *mut u16,
    mut q: DisjointSlice<u16>,
    mut k: DisjointSlice<u16>,
    mut v: DisjointSlice<u16>,
    mut kneg: DisjointSlice<u16>,
    mut kpos: DisjointSlice<f32>,
    mut chunk_g_last: DisjointSlice<f32>,
    q_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    k_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    g_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    beta_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_BETA_ELEMS>,
    params: CausalAttentionParams,
) {
    let bh = thread::blockIdx_x();
    let chunk = thread::blockIdx_y();
    let start = chunk * params.chunk_size;
    if bh >= batch_head(&params)
        || start >= params.seq_len
        || params.head_dim != 64
        || params.chunk_size > 64
    {
        return;
    }

    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    let end = params.seq_len.min(start + params.chunk_size);
    let chunk_tokens = end - start;
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_in_block = tid / 32;
    let warps_per_block = thread::blockDim_x() / 32;

    let mut token_in_chunk = warp_in_block;
    while token_in_chunk < chunk_tokens {
        let token = start + token_in_chunk;
        let row = batch * params.seq_len + token;
        let mut norm = KdaQkNormAcc::zero();
        let qk0 = read_qk_act(qkv, row, head, lane, &params);
        norm.add(qk0);
        let qk1 = read_qk_act(qkv, row, head, lane + 32, &params);
        norm.add(qk1);
        let (q_norm, k_norm) = norm.norms();
        let q_inv = params.scale / safe_denom(q_norm);
        let k_inv = 1.0 / safe_denom(k_norm);
        let beta_qkv_index = beta_index(row, head, &params);
        let raw_beta = if lane == 0 { qkv[beta_qkv_index] } else { 0.0 };
        let beta_value = warp::shuffle_f32(if lane == 0 { sigmoid(raw_beta) } else { 0.0 }, 0);
        if lane == 0 {
            beta_tile[token_in_chunk as usize] = beta_value;
            if !qkv_f16.is_null() {
                unsafe {
                    *qkv_f16.add(beta_qkv_index) = cvt_rn_f16_f32(raw_beta);
                }
            }
        }

        store_shared_prepare_dim(
            qkv,
            qkv_f16,
            q_tile,
            k_tile,
            g_tile,
            &mut v,
            batch,
            token,
            row,
            head,
            token_in_chunk,
            lane,
            qk0,
            q_inv,
            k_inv,
            beta_value,
            &params,
        );
        store_shared_prepare_dim(
            qkv,
            qkv_f16,
            q_tile,
            k_tile,
            g_tile,
            &mut v,
            batch,
            token,
            row,
            head,
            token_in_chunk,
            lane + 32,
            qk1,
            q_inv,
            k_inv,
            beta_value,
            &params,
        );

        let padding_index = head * 32 + lane;
        let active_qkv = beta_offset(&params) + params.head_count;
        if !qkv_f16.is_null() && active_qkv + padding_index < params.qkv_dim {
            let index = (row * params.qkv_dim + active_qkv + padding_index) as usize;
            unsafe {
                *qkv_f16.add(index) = cvt_rn_f16_f32(qkv[index]);
            }
        }
        token_in_chunk += warps_per_block;
    }
    thread::sync_threads();

    if tid < params.head_dim {
        let mut acc = 0.0;
        let mut prefix_token = 0;
        while prefix_token < chunk_tokens {
            let tile = (prefix_token * params.head_dim + tid) as usize;
            acc += g_tile[tile];
            g_tile[tile] = acc;
            prefix_token += 1;
        }
        unsafe {
            *chunk_g_last.get_unchecked_mut(chunk_g_last_index(bh, chunk, tid, &params)) = acc;
        }
    }
    thread::sync_threads();

    let mut transform_token = warp_in_block;
    while transform_token < chunk_tokens {
        let token = start + transform_token;
        store_shared_transform_dim(
            &mut q,
            &mut k,
            &mut kneg,
            &mut kpos,
            q_tile,
            k_tile,
            g_tile,
            beta_tile,
            batch,
            token,
            head,
            transform_token,
            chunk_tokens,
            lane,
            &params,
        );
        store_shared_transform_dim(
            &mut q,
            &mut k,
            &mut kneg,
            &mut kpos,
            q_tile,
            k_tile,
            g_tile,
            beta_tile,
            batch,
            token,
            head,
            transform_token,
            chunk_tokens,
            lane + 32,
            &params,
        );
        transform_token += warps_per_block;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_shared_prepare_dim(
    qkv: &[f32],
    qkv_f16: *mut u16,
    q_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    k_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    g_tile: &mut SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    v: &mut DisjointSlice<u16>,
    batch: u32,
    token: u32,
    row: u32,
    head: u32,
    token_in_chunk: u32,
    dim: u32,
    qk: KdaQkAct,
    q_inv: f32,
    k_inv: f32,
    beta_value: f32,
    params: &CausalAttentionParams,
) {
    let tile = (token_in_chunk * params.head_dim + dim) as usize;
    let v_index = qkv_index(row, head, dim, v_offset(params), params);
    let g_index = qkv_index(row, head, dim, g_offset(params), params);
    let raw_v = qkv[v_index];
    let raw_g = qkv[g_index];
    unsafe {
        q_tile[tile] = qk.q_act() * q_inv;
        k_tile[tile] = qk.k_act() * k_inv;
        g_tile[tile] = -params.decay_scale * softplus(raw_g);
        *v.get_unchecked_mut(compact_index(batch, token, head, dim, params)) =
            cvt_rn_f16_f32(value_activation(raw_v) * beta_value);
    }
    if !qkv_f16.is_null() {
        unsafe {
            *qkv_f16.add(qkv_index(row, head, dim, q_offset(params), params)) =
                cvt_rn_f16_f32(qk.raw_q);
            *qkv_f16.add(qkv_index(row, head, dim, k_offset(params), params)) =
                cvt_rn_f16_f32(qk.raw_k);
            *qkv_f16.add(v_index) = cvt_rn_f16_f32(raw_v);
            *qkv_f16.add(g_index) = cvt_rn_f16_f32(raw_g);
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_shared_transform_dim(
    q: &mut DisjointSlice<u16>,
    k: &mut DisjointSlice<u16>,
    kneg: &mut DisjointSlice<u16>,
    kpos: &mut DisjointSlice<f32>,
    q_tile: &SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    k_tile: &SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    g_tile: &SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS>,
    beta_tile: &SharedArray<f32, KDA_FUSED_PREPARE_BETA_ELEMS>,
    batch: u32,
    token: u32,
    head: u32,
    token_in_chunk: u32,
    chunk_tokens: u32,
    dim: u32,
    params: &CausalAttentionParams,
) {
    let tile = (token_in_chunk * params.head_dim + dim) as usize;
    let last_tile = ((chunk_tokens - 1) * params.head_dim + dim) as usize;
    let compact = compact_index(batch, token, head, dim, params);
    let q_value = q_tile[tile];
    let k_value = k_tile[tile];
    let g_value = g_tile[tile];
    let g_last = g_tile[last_tile];
    let beta_value = beta_tile[token_in_chunk as usize];
    let exp_g = kda_decay_exp(g_value);
    unsafe {
        *q.get_unchecked_mut(compact) = cvt_rn_f16_f32(q_value * exp_g);
        *kneg.get_unchecked_mut(compact) = cvt_rn_f16_f32(k_value * kda_decay_exp(-g_value));
        let kg_value = k_value * kda_decay_exp(g_last - g_value);
        *k.get_unchecked_mut(compact) = cvt_rn_f16_f32(kg_value);
        *(kpos.as_mut_ptr() as *mut u16).add(crate::kda_common::kpos_half_arena_index(compact)) =
            cvt_rn_f16_f32(k_value * beta_value * exp_g);
    }
}

pub(in super::super) fn zero_f32_body(mut values: DisjointSlice<f32>, element_count: u32) {
    let Some(index) = thread_index(element_count) else {
        return;
    };
    unsafe {
        *values.get_unchecked_mut(index as usize) = 0.0;
    }
}
