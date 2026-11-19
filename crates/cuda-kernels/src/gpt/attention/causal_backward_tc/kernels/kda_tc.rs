use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, launch_bounds, thread, warp};

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::super::kda::{
    KDA_INTRA_DG_TILE_ELEMS, KDA_INTRA_DKG_KG_TILE_ELEMS, KDA_INTRA_PAIR_STAT_ELEMS,
    KDA_INTRA_PAIR_VALUE_ELEMS, KDA_INTRA_THREADS_PER_BLOCK, KdaChunkwiseGrads, KdaChunkwiseInputs,
    KdaIntraGrads, KdaIntraInputs, NATIVE_STATE_TILE_ELEMS, chunk_intra_kda_backward_body,
    chunk_kda_dkg_from_vnew_dh_body, chunk_state_dw_dqg_matmul_body,
    chunk_state_project_du_dw_dqg_body, chunk_state_vnew_body, chunkwise_kda_backward_body,
    project_kda_intra_tiles,
};
use crate::attention::CausalAttentionParams;
use crate::block_reduce::block_max_leader_f32;
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};

const KDA_INTRA_WARPS_PER_BLOCK: usize = (KDA_INTRA_THREADS_PER_BLOCK / 32) as usize;
const KDA_INTRA_AMAX_OUTPUTS: u32 = TC_BACKWARD_THREADS_PER_BLOCK / 32;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn chunk_kda_vnew_from_state_kernel(
        w: &[u16],
        u: &[f32],
        chunk_states: &[u16],
        v_new: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_vnew_body; w, u, chunk_states, v_new, params);
    }

    #[kernel]
    pub fn chunk_kda_dw_dqg_from_state_kernel(
        d_u: &[u16],
        d_out_compact: &[u16],
        chunk_states: &[u16],
        d_w: DisjointSlice<u16>,
        d_qg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_dw_dqg_matmul_body; d_u, d_out_compact, chunk_states, d_w, d_qg, params);
    }

    #[kernel]
    pub fn chunk_kda_project_du_dw_dqg_kernel(
        aqk: &[u16],
        d_out: &[u16],
        chunk_states: &[u16],
        d_u: DisjointSlice<u16>,
        d_w: DisjointSlice<u16>,
        d_qg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut DU: SharedArray<u16, NATIVE_STATE_TILE_ELEMS> = SharedArray::UNINIT;
        static mut DOUT: SharedArray<u16, NATIVE_STATE_TILE_ELEMS> = SharedArray::UNINIT;
        with_tc_ab_tiles!(chunk_state_project_du_dw_dqg_body; aqk, d_out, chunk_states, d_u, d_w, d_qg, params; unsafe { &mut DU }, unsafe { &mut DOUT });
    }

    #[kernel]
    pub fn chunk_kda_dkg_from_vnew_dh_kernel(
        v_new: &[u16],
        d_h_states: &[u16],
        d_kg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_kda_dkg_from_vnew_dh_body; v_new, d_h_states, d_kg, params);
    }

    #[kernel]
    pub fn chunkwise_kda_backward_kernel(
        qg: &[f32],
        kg: &[f32],
        d_u: DisjointSlice<u16>,
        w: &[u16],
        g: &[f32],
        aqk: &[u16],
        d_out: &[u16],
        d_h_states: DisjointSlice<u16>,
        params: CausalAttentionParams,
    ) {
        let inputs = KdaChunkwiseInputs {
            qg,
            kg,
            w,
            g,
            aqk,
            d_out,
        };
        let grads = KdaChunkwiseGrads { d_u, d_h_states };
        with_kda_tiles!(backward chunkwise_kda_backward_body; inputs, grads, params);
    }

    #[kernel]
    #[launch_bounds(512, 2)]
    pub fn chunk_intra_kda_backward_kernel(
        qkv: &[u16],
        qg: &[f32],
        kg: &[f32],
        vbeta: &[f32],
        g: &[f32],
        beta: &[f32],
        d_qg: &[f32],
        d_kg: &[f32],
        d_k_a: &[f32],
        d_kpos_m: &[f32],
        d_vbeta_m: &[f32],
        inverse: &[u16],
        d_u: &[u16],
        d_w: &[u16],
        kneg: &[u16],
        kpos: &[u16],
        mut q_norms: DisjointSlice<f32>,
        mut k_norms: DisjointSlice<f32>,
        mut d_qkv: DisjointSlice<f32>,
        mut d_qkv_chunk_amax: DisjointSlice<f32>,
        accumulate_value_grad: u32,
        detached_chunk_state: u32,
        params: CausalAttentionParams,
    ) {
        static mut INTRA_AMAX: SharedArray<f32, KDA_INTRA_WARPS_PER_BLOCK> = SharedArray::UNINIT;
        static mut DG_TILE: SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS> = SharedArray::UNINIT;
        static mut DKG_KG_TILE: SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS> = SharedArray::UNINIT;
        static mut PAIR_VALUES: SharedArray<f32, KDA_INTRA_PAIR_VALUE_ELEMS> = SharedArray::UNINIT;
        static mut PAIR_STATS: SharedArray<f32, KDA_INTRA_PAIR_STAT_ELEMS> = SharedArray::UNINIT;
        static mut INTRA_PARAMS: SharedArray<CausalAttentionParams, 1> = SharedArray::UNINIT;
        static mut BETA_CHUNK: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut G_LAST_CHUNK: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut PROJECTION_A: crate::kda_tc::CtaATile = SharedArray::UNINIT;
        static mut PROJECTION_B: crate::kda_tc::CtaBTile = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        if tid == 0 {
            unsafe { INTRA_PARAMS[0] = params };
        }
        if tid < 64 {
            // The host requires complete 64-token chunks. Each cached scalar
            // keeps its original FP32 bits and is reused by all 64 dimensions
            // or tokens behind the already-required parameter barrier.
            let start = thread::blockIdx_x() * params.seq_len + thread::blockIdx_y() * 64;
            unsafe {
                BETA_CHUNK[tid as usize] = beta[(start + tid) as usize];
                G_LAST_CHUNK[tid as usize] = g[((start + 63) * 64 + tid) as usize];
            }
        }
        thread::sync_threads();
        let params = unsafe { &INTRA_PARAMS[0] };
        project_kda_intra_tiles!(
            inverse,
            kneg,
            kpos,
            vbeta,
            d_u,
            d_w,
            unsafe { &mut DG_TILE },
            unsafe { &mut DKG_KG_TILE },
            unsafe { &mut PROJECTION_A },
            unsafe { &mut PROJECTION_B },
            params
        );

        let inputs = KdaIntraInputs {
            qkv,
            qg,
            kg,
            vbeta,
            g,
            beta_chunk: unsafe { &BETA_CHUNK },
            g_last_chunk: unsafe { &G_LAST_CHUNK },
            detached_chunk_state,
            d_kg,
            d_kpos_m,
            d_vbeta_m,
            d_qg,
            d_k_a,
        };
        let grads = KdaIntraGrads {
            q_norms: q_norms.as_mut_ptr(),
            k_norms: k_norms.as_mut_ptr(),
            d_qkv: d_qkv.as_mut_ptr(),
        };
        let local_amax = unsafe {
            chunk_intra_kda_backward_body(
                inputs,
                grads,
                &mut DG_TILE,
                &mut DKG_KG_TILE,
                &mut PAIR_VALUES,
                &mut PAIR_STATS,
                accumulate_value_grad,
                params,
            )
        };
        let lane = warp::lane_id();
        let warp_in_block = tid / 32;
        let block_amax =
            unsafe { block_max_leader_f32(&mut INTRA_AMAX, local_amax, lane, warp_in_block) };
        if tid < KDA_INTRA_AMAX_OUTPUTS {
            let chunk = thread::blockIdx_x() * thread::gridDim_y() + thread::blockIdx_y();
            let value = if tid == 0 {
                block_amax.unwrap_or(0.0)
            } else {
                0.0
            };
            unsafe {
                *d_qkv_chunk_amax
                    .get_unchecked_mut((chunk * KDA_INTRA_AMAX_OUTPUTS + tid) as usize) = value;
            }
        }
    }
}

pub(super) use module::{LoadedModule, from_module};
