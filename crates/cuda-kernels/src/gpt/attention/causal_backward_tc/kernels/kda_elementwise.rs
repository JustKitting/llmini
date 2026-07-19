use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::super::kda::{
    FinishKdaGrads, add_kda_compact_body, chunk_cumsum_g_body, finish_kda_backward_body,
    gather_kda_dout_body, make_kda_kneg_from_kg_body, make_kda_kneg_kpos_from_kg_body,
    make_kda_kpos_from_kg_body, make_kda_strict_neg_matrix_body, prepare_kda_backward_inputs_body,
};
use crate::attention::CausalAttentionParams;
use crate::block_reduce::{block_max_pair_leader_f32, block_max_store_f32};
use crate::warp_reduce::thread_lane_warp;

const NORM_REDUCE_THREADS_PER_BLOCK: u32 = 256;
const NORM_REDUCE_WARPS_PER_BLOCK: usize = (NORM_REDUCE_THREADS_PER_BLOCK / 32) as usize;
const FINISH_WARPS_PER_BLOCK: usize = (TC_BACKWARD_THREADS_PER_BLOCK / 32) as usize;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn prepare_kda_backward_inputs_kernel(
        qkv: &[u16],
        q: DisjointSlice<f32>,
        k: DisjointSlice<f32>,
        v: DisjointSlice<f32>,
        g: DisjointSlice<f32>,
        beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        prepare_kda_backward_inputs_body(qkv, q, k, v, g, beta, params);
    }

    #[kernel]
    pub fn chunk_cumsum_kda_backward_g_kernel(
        g: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        chunk_cumsum_g_body(g, params);
    }

    #[kernel]
    pub fn gather_kda_dout_kernel(
        d_out: &[f32],
        compact_out: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        gather_kda_dout_body(d_out, compact_out, params);
    }

    #[kernel]
    pub fn add_kda_compact_kernel(
        dst: DisjointSlice<f32>,
        src: &[f32],
        params: CausalAttentionParams,
    ) {
        add_kda_compact_body(dst, src, params);
    }

    #[kernel]
    pub fn make_kda_backward_kneg_from_kg_kernel(
        kg: &[f32],
        g: &[f32],
        kneg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kneg_from_kg_body(kg, g, kneg, params);
    }

    #[kernel]
    pub fn make_kda_backward_kpos_from_kg_kernel(
        kg: &[f32],
        g: &[f32],
        beta: &[f32],
        kpos: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kpos_from_kg_body(kg, g, beta, kpos, params);
    }

    #[kernel]
    pub fn make_kda_backward_kneg_kpos_from_kg_kernel(
        kg: &[f32],
        g: &[f32],
        beta: &[f32],
        kneg: DisjointSlice<f32>,
        kpos: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_kneg_kpos_from_kg_body(kg, g, beta, kneg, kpos, params);
    }

    #[kernel]
    pub fn make_kda_strict_neg_matrix_kernel(
        src: &[f32],
        dst: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        make_kda_strict_neg_matrix_body(src, dst, params);
    }

    #[kernel]
    pub fn finish_kda_backward_kernel(
        qkv: &[u16],
        d_q: &[f32],
        d_k: &[f32],
        d_v: &[f32],
        d_g: &[f32],
        d_beta: &[f32],
        q_norms: DisjointSlice<f32>,
        k_norms: DisjointSlice<f32>,
        d_qkv: DisjointSlice<f32>,
        mut d_qkv_chunk_amax: DisjointSlice<f32>,
        accumulate_value_grad: u32,
        params: CausalAttentionParams,
    ) {
        static mut FINISH_AMAX: SharedArray<f32, FINISH_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let local_amax = finish_kda_backward_body(
            qkv,
            FinishKdaGrads {
                q: d_q,
                k: d_k,
                v: d_v,
                g: d_g,
                beta: d_beta,
            },
            q_norms,
            k_norms,
            d_qkv,
            accumulate_value_grad,
            params,
        );
        let (_, lane, warp_in_block) = thread_lane_warp();
        block_max_store_f32!(
            FINISH_AMAX,
            d_qkv_chunk_amax[thread::blockIdx_x()],
            local_amax,
            lane,
            warp_in_block
        );
    }

    #[kernel]
    pub fn reduce_kda_qk_norm_max_kernel(
        q_norms: &[f32],
        k_norms: &[f32],
        mut qk_norm_max: DisjointSlice<f32>,
        norm_offset: u32,
        row_count: u32,
        head_count: u32,
    ) {
        let head = thread::blockIdx_x();
        if head >= head_count {
            return;
        }

        static mut REDUCE: SharedArray<f32, NORM_REDUCE_WARPS_PER_BLOCK> = SharedArray::UNINIT;
        static mut REDUCE_PAIR: SharedArray<f32, NORM_REDUCE_WARPS_PER_BLOCK> = SharedArray::UNINIT;
        let (tid, lane, warp_id) = thread_lane_warp();
        let mut q_max = 0.0;
        let mut k_max = 0.0;
        let mut row = tid;
        while row < row_count {
            let index = (head * row_count + row) as usize;
            let q_norm = q_norms[index];
            let k_norm = k_norms[index];
            q_max = if q_norm > q_max { q_norm } else { q_max };
            k_max = if k_norm > k_max { k_norm } else { k_max };
            row += NORM_REDUCE_THREADS_PER_BLOCK;
        }

        if let Some((q_max, k_max)) = unsafe {
            block_max_pair_leader_f32(&mut REDUCE, &mut REDUCE_PAIR, q_max, k_max, lane, warp_id)
        } {
            unsafe {
                *qk_norm_max.get_unchecked_mut((norm_offset + head) as usize) = q_max;
                *qk_norm_max.get_unchecked_mut((norm_offset + head_count + head) as usize) = k_max;
            }
        }
    }
}

pub(super) use module::{LoadedModule, from_module};
pub(crate) const KDA_NORM_REDUCE_THREADS_PER_BLOCK: u32 = NORM_REDUCE_THREADS_PER_BLOCK;
