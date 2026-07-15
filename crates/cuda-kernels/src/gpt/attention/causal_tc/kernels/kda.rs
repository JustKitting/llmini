use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::super::kda::{
    KdaStateSaveInputs, chunk_cumsum_g_body, chunk_kda_output_from_state_body,
    chunk_kda_state_save_body, make_kg_kpos_vbeta_body, make_kneg_from_kg_body,
    make_qg_kg_vbeta_body, make_qg_kneg_body, make_qg_kneg_kg_kpos_vbeta_body, mask_akk_body,
    mask_aqk_body, prepare_kda_body, solve_akk_inv_body, store_chunk_g_last_body, zero_f32_body,
};
use crate::attention::CausalAttentionParams;
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn prepare_kda_forward_kernel(
        qkv: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_kda_body(qkv, core::ptr::null_mut(), q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn prepare_kda_forward_save_f16_kernel(
        qkv: &[f32],
        mut qkv_f16: DisjointSlice<u16>,
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_kda_body(qkv, qkv_f16.as_mut_ptr(), q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn zero_kda_f32_kernel(values: DisjointSlice<f32>, element_count: u32) {
        zero_f32_body(values, element_count);
    }

    #[kernel]
    pub fn chunk_cumsum_kda_g_kernel(g: DisjointSlice<f32>, params: CausalAttentionParams) {
        chunk_cumsum_g_body(g, params);
    }

    #[kernel]
    pub fn make_kda_qg_kneg_kernel(
        q: DisjointSlice<f32>,
        k: &[f32],
        g: &[f32],
        kneg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_qg_kneg_body(q, k, g, kneg, params);
    }

    #[kernel]
    pub fn make_kda_kg_kpos_vbeta_kernel(
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: &[f32],
        beta: &[f32],
        kpos_beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kg_kpos_vbeta_body(k, v, g, beta, kpos_beta, params);
    }

    #[kernel]
    pub fn make_kda_qg_kneg_kg_kpos_vbeta_kernel(
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: &[f32],
        beta: &[f32],
        kneg: DisjointSlice<f32>,
        kpos_beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_qg_kneg_kg_kpos_vbeta_body(q, k, v, g, beta, kneg, kpos_beta, params);
    }

    #[kernel]
    pub fn make_kda_qg_kg_vbeta_kernel(
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: &[f32],
        beta: &[f32],
        params: CausalAttentionParams,
    ) {
        make_qg_kg_vbeta_body(q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn store_kda_chunk_g_last_kernel(
        g: &[f32],
        chunk_g_last: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        store_chunk_g_last_body(g, chunk_g_last, params);
    }

    #[kernel]
    pub fn make_kda_kneg_from_kg_kernel(
        k: &[f32],
        chunk_g_last: &[f32],
        kneg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kneg_from_kg_body(k, chunk_g_last, kneg, params);
    }

    #[kernel]
    pub fn mask_kda_aqk_kernel(aqk: DisjointSlice<f32>, params: CausalAttentionParams) {
        mask_aqk_body(aqk, params);
    }

    #[kernel]
    pub fn mask_kda_akk_kernel(akk: DisjointSlice<f32>, params: CausalAttentionParams) {
        mask_akk_body(akk, params);
    }

    #[kernel]
    pub fn solve_kda_akk_inv_kernel(akk: DisjointSlice<f32>, params: CausalAttentionParams) {
        with_kda_tiles!(inv solve_akk_inv_body; akk, params);
    }

    #[kernel]
    pub fn chunk_kda_state_save_kernel(
        kg: &[f32],
        v_new: DisjointSlice<f32>,
        w: &[f32],
        u: &[f32],
        chunk_g_last: &[f32],
        chunk_states: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state chunk_kda_state_save_body; KdaStateSaveInputs { kg, w, u, chunk_g_last }, v_new, chunk_states, params);
    }

    #[kernel]
    pub fn chunk_kda_output_from_state_kernel(
        qg: &[f32],
        v_new: &[f32],
        aqk: &[f32],
        out: DisjointSlice<f32>,
        chunk_states: &[u16],
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_kda_output_from_state_body; qg, v_new, aqk, out, chunk_states, params);
    }
}

pub(in crate::attention) use module::{LoadedModule, from_module};
