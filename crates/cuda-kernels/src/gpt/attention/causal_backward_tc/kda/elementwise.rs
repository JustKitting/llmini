use cuda_device::{DisjointSlice, SharedArray, thread, warp};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::kda_common::{
    batch_head, beta_compact_index, beta_index, chunk_count, chunk_matrix_elems, compact_elems,
    compact_index, compact_linear_parts, g_offset, hidden_index, kda_decay_exp,
    linear_thread_index, qkv_index, safe_denom, sigmoid, softplus, v_offset, value_activation,
};
use crate::kda_elementwise::{
    KdaPrepareOutputs, KdaQkAct, KdaQkNormAcc, chunk_cumsum_g_body as shared_chunk_cumsum_g_body,
    prepare_kda_inputs_body, read_qk_act,
};

pub(in super::super) const KDA_FUSED_BACKWARD_TILE_ELEMS: usize = 64 * 64;
pub(in super::super) const KDA_FUSED_BACKWARD_BETA_ELEMS: usize = 64;

pub(crate) fn prepare_kda_backward_inputs_body(
    qkv: &[u16],
    q: DisjointSlice<f32>,
    k: DisjointSlice<f32>,
    v: DisjointSlice<f32>,
    g: DisjointSlice<f32>,
    beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    prepare_kda_inputs_body(
        qkv,
        KdaPrepareOutputs { q, k, v, g, beta },
        core::ptr::null_mut(),
        params,
        TC_BACKWARD_THREADS_PER_BLOCK,
    );
}

pub(crate) fn chunk_cumsum_g_body(g: DisjointSlice<f32>, params: CausalAttentionParams) {
    shared_chunk_cumsum_g_body(g, params);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_transform_kda_backward_cached_body(
    qkv: &[u16],
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    mut g: DisjointSlice<f32>,
    mut beta: DisjointSlice<f32>,
    mut native_kpos: DisjointSlice<u16>,
    mut native_kneg: DisjointSlice<u16>,
    q_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    k_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    g_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    beta_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_BETA_ELEMS>,
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
        let raw_beta = if lane == 0 {
            cvt_f32_f16(qkv[beta_qkv_index])
        } else {
            0.0
        };
        let beta_value = warp::shuffle_f32(if lane == 0 { sigmoid(raw_beta) } else { 0.0 }, 0);
        if lane == 0 {
            beta_tile[token_in_chunk as usize] = beta_value;
            unsafe {
                *beta.get_unchecked_mut(beta_compact_index(batch, token, head, &params)) =
                    beta_value;
            }
        }

        store_backward_prepare_dim(
            qkv,
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
        store_backward_prepare_dim(
            qkv,
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
        token_in_chunk += warps_per_block;
    }
    thread::sync_threads();

    if tid < params.head_dim {
        let mut acc = 0.0;
        let mut token_in_chunk = 0;
        while token_in_chunk < chunk_tokens {
            let tile = (token_in_chunk * params.head_dim + tid) as usize;
            acc += g_tile[tile];
            g_tile[tile] = acc;
            unsafe {
                *g.get_unchecked_mut(compact_index(
                    batch,
                    start + token_in_chunk,
                    head,
                    tid,
                    &params,
                )) = acc;
            }
            token_in_chunk += 1;
        }
    }
    thread::sync_threads();

    let mut token_in_chunk = warp_in_block;
    while token_in_chunk < chunk_tokens {
        let token = start + token_in_chunk;
        store_backward_transform_dim(
            &mut q,
            &mut k,
            &mut native_kpos,
            &mut native_kneg,
            beta_tile[token_in_chunk as usize],
            q_tile,
            k_tile,
            g_tile,
            batch,
            token,
            head,
            token_in_chunk,
            chunk_tokens,
            lane,
            &params,
        );
        store_backward_transform_dim(
            &mut q,
            &mut k,
            &mut native_kpos,
            &mut native_kneg,
            beta_tile[token_in_chunk as usize],
            q_tile,
            k_tile,
            g_tile,
            batch,
            token,
            head,
            token_in_chunk,
            chunk_tokens,
            lane + 32,
            &params,
        );
        token_in_chunk += warps_per_block;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_backward_prepare_dim(
    qkv: &[u16],
    q_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    k_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    g_tile: &mut SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    v: &mut DisjointSlice<f32>,
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
    let raw_v = cvt_f32_f16(qkv[qkv_index(row, head, dim, v_offset(params), params)]);
    let raw_g = cvt_f32_f16(qkv[qkv_index(row, head, dim, g_offset(params), params)]);
    unsafe {
        q_tile[tile] = qk.q_act() * q_inv;
        k_tile[tile] = qk.k_act() * k_inv;
        g_tile[tile] = -params.decay_scale * softplus(raw_g);
        *v.get_unchecked_mut(compact_index(batch, token, head, dim, params)) =
            value_activation(raw_v) * beta_value;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_backward_transform_dim(
    q: &mut DisjointSlice<f32>,
    k: &mut DisjointSlice<f32>,
    native_kpos: &mut DisjointSlice<u16>,
    native_kneg: &mut DisjointSlice<u16>,
    beta_value: f32,
    q_tile: &SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    k_tile: &SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
    g_tile: &SharedArray<f32, KDA_FUSED_BACKWARD_TILE_ELEMS>,
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
    let g_value = g_tile[tile];
    let g_last = g_tile[last_tile];
    let kg_value = k_tile[tile] * kda_decay_exp(g_last - g_value);
    unsafe {
        *q.get_unchecked_mut(compact) = q_tile[tile] * kda_decay_exp(g_value);
        *k.get_unchecked_mut(compact) = kg_value;
        // Match forward's direct normalized-K factors. Reconstructing them
        // from Kg with summed exponents is invalid once either clamp engages.
        *native_kpos.get_unchecked_mut(compact) = crate::f16_tc_matmul::convert::cvt_rn_f16_f32(
            k_tile[tile] * beta_value * kda_decay_exp(g_value),
        );
        *native_kneg.get_unchecked_mut(compact) =
            crate::f16_tc_matmul::convert::cvt_rn_f16_f32(k_tile[tile] * kda_decay_exp(-g_value));
    }
}

pub(crate) fn gather_kda_dout_body(
    d_out: &[f32],
    mut compact_out: DisjointSlice<u16>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    unsafe {
        *compact_out.get_unchecked_mut(index as usize) =
            crate::f16_tc_matmul::convert::cvt_rn_f16_f32(
                d_out[hidden_index(batch, token, head, dim, &params)],
            );
    }
}

pub(crate) fn add_kda_compact_body(
    mut dst: DisjointSlice<f32>,
    src: &[f32],
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    unsafe {
        *dst.get_unchecked_mut(index as usize) += src[index as usize];
    }
}

pub(crate) fn make_kda_strict_neg_matrix_body(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let chunks = chunk_count(&params);
    let matrix_elems = chunk_matrix_elems(&params);
    let total = batch_head(&params) * chunks * matrix_elems;
    let Some(index) = thread_index(total) else {
        return;
    };
    let elem = index % matrix_elems;
    let row = elem / params.chunk_size;
    let col = elem - row * params.chunk_size;
    let value = if row > col { -src[index as usize] } else { 0.0 };
    unsafe {
        *dst.get_unchecked_mut(index as usize) = value;
    }
}

#[inline(always)]
fn thread_index(element_count: u32) -> Option<u32> {
    linear_thread_index(TC_BACKWARD_THREADS_PER_BLOCK, element_count)
}
