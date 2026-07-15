use cuda_device::DisjointSlice;

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::attention::CausalAttentionParams;
use crate::kda_common::{
    batch_head, beta_compact_index, chunk_count, chunk_end_token, chunk_matrix_elems,
    compact_elems, compact_index, compact_linear_parts, hidden_index, kda_decay_exp,
    linear_thread_index,
};
use crate::kda_elementwise::{
    KdaPrepareOutputs, chunk_cumsum_g_body as shared_chunk_cumsum_g_body, prepare_kda_inputs_body,
};

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

pub(crate) fn gather_kda_dout_body(
    d_out: &[f32],
    mut compact_out: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    unsafe {
        *compact_out.get_unchecked_mut(index as usize) =
            d_out[hidden_index(batch, token, head, dim, &params)];
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

pub(crate) fn make_kda_kneg_from_kg_body(
    kg: &[f32],
    g: &[f32],
    mut kneg: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    unsafe {
        *kneg.get_unchecked_mut(index as usize) = kg[index as usize] * kda_decay_exp(-g_last);
    }
}

pub(crate) fn make_kda_kpos_from_kg_body(
    kg: &[f32],
    g: &[f32],
    beta: &[f32],
    mut kpos: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let compact = compact_index(batch, token, head, dim, &params);
    let g_value = g[compact];
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    let beta_value = beta[beta_compact_index(batch, token, head, &params)];
    unsafe {
        *kpos.get_unchecked_mut(index as usize) =
            beta_value * kg[index as usize] * kda_decay_exp(2.0 * g_value - g_last);
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
