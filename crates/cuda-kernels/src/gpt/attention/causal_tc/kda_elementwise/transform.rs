use cuda_device::DisjointSlice;

use super::thread_index;
use crate::attention::CausalAttentionParams;
use crate::kda_common::{
    beta_compact_index, chunk_end_token, chunk_g_last_index, compact_elems, compact_index,
    compact_linear_parts, kda_decay_exp,
};

pub(in super::super) fn make_qg_kneg_body(
    mut q: DisjointSlice<f32>,
    k: &[f32],
    g: &[f32],
    mut kneg: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let value_g = g[index as usize];
    unsafe {
        *q.get_unchecked_mut(index as usize) *= kda_decay_exp(value_g);
        *kneg.get_unchecked_mut(index as usize) = k[index as usize] * kda_decay_exp(-value_g);
    }
}

pub(in super::super) fn make_kg_kpos_vbeta_body(
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    g: &[f32],
    beta: &[f32],
    mut kpos_beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let g_value = g[index as usize];
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    let beta_value = beta[beta_compact_index(batch, token, head, &params)];

    unsafe {
        let k_value = *k.get_unchecked_mut(index as usize);
        *k.get_unchecked_mut(index as usize) = k_value * kda_decay_exp(g_last - g_value);
        *v.get_unchecked_mut(index as usize) *= beta_value;
        *kpos_beta.get_unchecked_mut(index as usize) =
            k_value * beta_value * kda_decay_exp(g_value);
    }
}

pub(in super::super) fn make_qg_kneg_kg_kpos_vbeta_body(
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    g: &[f32],
    beta: &[f32],
    mut kneg: DisjointSlice<f32>,
    mut kpos_beta: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let compact = index as usize;
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let g_value = g[compact];
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    let beta_value = beta[beta_compact_index(batch, token, head, &params)];
    let exp_g = kda_decay_exp(g_value);

    unsafe {
        let q_value = *q.get_unchecked_mut(compact);
        let k_value = *k.get_unchecked_mut(compact);
        let v_value = *v.get_unchecked_mut(compact);
        *q.get_unchecked_mut(compact) = q_value * exp_g;
        *kneg.get_unchecked_mut(compact) = k_value * kda_decay_exp(-g_value);
        *k.get_unchecked_mut(compact) = k_value * kda_decay_exp(g_last - g_value);
        *v.get_unchecked_mut(compact) = v_value * beta_value;
        *kpos_beta.get_unchecked_mut(compact) = k_value * beta_value * exp_g;
    }
}

pub(in super::super) fn make_qg_kg_vbeta_body(
    mut q: DisjointSlice<f32>,
    mut k: DisjointSlice<f32>,
    mut v: DisjointSlice<f32>,
    g: &[f32],
    beta: &[f32],
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, _bh, batch, head) = compact_linear_parts(index, &params);
    let compact = index as usize;
    let chunk_end = chunk_end_token(token / params.chunk_size, &params);
    let g_value = g[compact];
    let g_last = g[compact_index(batch, chunk_end, head, dim, &params)];
    let beta_value = beta[beta_compact_index(batch, token, head, &params)];

    unsafe {
        *q.get_unchecked_mut(compact) *= kda_decay_exp(g_value);
        *k.get_unchecked_mut(compact) *= kda_decay_exp(g_last - g_value);
        *v.get_unchecked_mut(compact) *= beta_value;
    }
}

pub(in super::super) fn make_kneg_from_kg_body(
    k: &[f32],
    chunk_g_last: &[f32],
    mut kneg: DisjointSlice<f32>,
    params: CausalAttentionParams,
) {
    let Some(index) = thread_index(compact_elems(&params)) else {
        return;
    };
    let (dim, token, bh, _batch, _head) = compact_linear_parts(index, &params);
    let chunk = token / params.chunk_size;
    let g_last = chunk_g_last[chunk_g_last_index(bh, chunk, dim, &params)];
    unsafe {
        *kneg.get_unchecked_mut(index as usize) = k[index as usize] * kda_decay_exp(-g_last);
    }
}
