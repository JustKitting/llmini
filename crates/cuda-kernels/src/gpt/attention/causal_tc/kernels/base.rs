use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};

use super::super::gather::{gather_qk_v_f16_body, gather_qknorm_v_f16_body, gather_qkv_body};
use super::super::scatter::{scatter_output_body, scatter_output_save_f16_body};
use super::super::softmax::{softmax_body, softmax_f16_body};
use crate::attention::CausalAttentionParams;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn gather_qkv_forward_kernel(
        qkv: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        gather_qkv_body(qkv, q, k, v, params);
    }

    #[kernel]
    pub fn gather_qknorm_v_f16_forward_kernel(
        qkv: &[f32],
        qk_scale_bytes: &[u8],
        qk_scale_scales: &[u8],
        qk_scale_global_scale: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        static mut Q_WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut K_WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut PREVIOUS_K_WARP_SUMS: SharedArray<f32, 8> = SharedArray::UNINIT;
        gather_qknorm_v_f16_body(
            qkv,
            qk_scale_bytes,
            qk_scale_scales,
            qk_scale_global_scale,
            q,
            k,
            v,
            params,
            unsafe { &mut Q_WARP_SUMS },
            unsafe { &mut K_WARP_SUMS },
            unsafe { &mut PREVIOUS_K_WARP_SUMS },
        );
    }

    #[kernel]
    pub fn gather_qk_v_f16_forward_kernel(
        qkv: &[f32],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        gather_qk_v_f16_body(qkv, q, k, v, params);
    }

    #[kernel]
    pub fn attention_softmax_forward_kernel(
        scores: &[f32],
        probs: DisjointSlice<f32>,
        log_sum_exp: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
        softmax_body(scores, probs, log_sum_exp, params, unsafe { &mut REDUCE });
    }

    #[kernel]
    pub fn attention_softmax_forward_f16_kernel(
        scores: &[f32],
        probs: DisjointSlice<u16>,
        log_sum_exp: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
        softmax_f16_body(scores, probs, log_sum_exp, params, unsafe { &mut REDUCE });
    }

    #[kernel]
    pub fn scatter_attention_forward_kernel(
        compact: &[f32],
        out: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        scatter_output_body(compact, out, params);
    }

    #[kernel]
    pub fn scatter_attention_forward_save_f16_kernel(
        compact: &[f32],
        out: DisjointSlice<f32>,
        attention_out_f16: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        scatter_output_save_f16_body(compact, out, attention_out_f16, params);
    }
}

pub(super) use module::{LoadedModule, from_module};
