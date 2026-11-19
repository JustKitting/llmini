use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::super::kda::{
    KDA_FUSED_PREPARE_BETA_ELEMS, KDA_FUSED_PREPARE_TILE_ELEMS, KdaStateSaveInputs,
    chunk_kda_output_from_state_body, chunk_kda_state_save_body, make_kg_kpos_vbeta_body,
    make_qg_kg_vbeta_body, make_qg_kneg_body, make_qg_kneg_kg_kpos_vbeta_body, mask_akk_body,
    mask_aqk_body, prepare_transform_kda_body, zero_f32_body,
};
use crate::attention::CausalAttentionParams;
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};

#[cuda_module]
pub(super) mod module {
    use super::*;

    macro_rules! prepare_native {
        ($qkv:expr, $tape:expr, $q:expr, $k:expr, $v:expr, $kneg:expr, $kpos:expr, $g:expr, $params:expr) => {{
            static mut Q_TILE: SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS> = SharedArray::UNINIT;
            static mut K_TILE: SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS> = SharedArray::UNINIT;
            static mut G_TILE: SharedArray<f32, KDA_FUSED_PREPARE_TILE_ELEMS> = SharedArray::UNINIT;
            static mut BETA_TILE: SharedArray<f32, KDA_FUSED_PREPARE_BETA_ELEMS> =
                SharedArray::UNINIT;
            unsafe {
                prepare_transform_kda_body(
                    $qkv,
                    $tape,
                    $q,
                    $k,
                    $v,
                    $kneg,
                    $kpos,
                    $g,
                    &mut Q_TILE,
                    &mut K_TILE,
                    &mut G_TILE,
                    &mut BETA_TILE,
                    $params,
                );
            }
        }};
    }

    #[kernel]
    pub fn prepare_transform_kda_forward_kernel(
        qkv: &[f32],
        q: DisjointSlice<u16>,
        k: DisjointSlice<u16>,
        v: DisjointSlice<u16>,
        kneg: DisjointSlice<u16>,
        kpos: DisjointSlice<f32>,
        chunk_g_last: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_native!(
            qkv,
            core::ptr::null_mut(),
            q,
            k,
            v,
            kneg,
            kpos,
            chunk_g_last,
            params
        );
    }

    #[kernel]
    pub fn prepare_transform_kda_forward_save_f16_kernel(
        qkv: &[f32],
        mut qkv_f16: DisjointSlice<u16>,
        q: DisjointSlice<u16>,
        k: DisjointSlice<u16>,
        v: DisjointSlice<u16>,
        kneg: DisjointSlice<u16>,
        kpos: DisjointSlice<f32>,
        chunk_g_last: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_native!(
            qkv,
            qkv_f16.as_mut_ptr(),
            q,
            k,
            v,
            kneg,
            kpos,
            chunk_g_last,
            params
        );
    }

    #[kernel]
    pub fn zero_kda_f32_kernel(values: DisjointSlice<f32>, element_count: u32) {
        zero_f32_body(values, element_count);
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
        native_kpos: DisjointSlice<u16>,
        native_kneg: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        make_qg_kneg_kg_kpos_vbeta_body(
            q,
            k,
            v,
            g,
            beta,
            kneg,
            kpos_beta,
            native_kpos,
            native_kneg,
            params,
        );
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
    pub fn mask_kda_aqk_kernel(aqk: DisjointSlice<f32>, params: CausalAttentionParams) {
        mask_aqk_body(aqk, params);
    }

    #[kernel]
    pub fn mask_kda_akk_kernel(akk: DisjointSlice<f32>, params: CausalAttentionParams) {
        mask_akk_body(akk, params);
    }

    #[kernel]
    pub fn chunk_kda_state_save_kernel(
        kg: &[u16],
        v_new: DisjointSlice<u16>,
        w: &[u16],
        u: &[f32],
        chunk_g_last: &[f32],
        chunk_states: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        with_kda_tiles!(state chunk_kda_state_save_body; KdaStateSaveInputs { kg, w, u, chunk_g_last }, v_new, chunk_states, params);
    }

    #[kernel]
    pub fn chunk_kda_output_from_state_kernel(
        qg: &[u16],
        v_new: &[u16],
        aqk: &[u16],
        out: DisjointSlice<f32>,
        chunk_states: &[u16],
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_kda_output_from_state_body; qg, v_new, aqk, out, chunk_states, params);
    }
}

pub(in crate::attention) use module::{LoadedModule, from_module};
