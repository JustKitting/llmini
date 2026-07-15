use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::super::kda::{
    ChunkStateMatmulMode, KdaChunkwiseGrads, KdaChunkwiseInputs, KdaDmInputs, KdaIntraGrads,
    KdaIntraInputs, chunk_intra_kda_backward_body, chunk_intra_kda_dm_body,
    chunk_kda_dkg_from_vnew_dh_body, chunk_state_matmul_body, chunkwise_kda_backward_body,
};
use crate::attention::CausalAttentionParams;
use crate::kda_tc::{with_kda_tiles, with_tc_ab_tiles};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn chunk_kda_vnew_from_state_kernel(
        w: &[f32],
        u: &[f32],
        chunk_states: &[u16],
        v_new: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; w, u, chunk_states, v_new, params; ChunkStateMatmulMode::VNew);
    }

    #[kernel]
    pub fn chunk_kda_dw_from_du_state_kernel(
        d_u: &[f32],
        chunk_states: &[u16],
        d_w: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; d_u, d_u, chunk_states, d_w, params; ChunkStateMatmulMode::Dw);
    }

    #[kernel]
    pub fn chunk_kda_dqg_from_dout_state_kernel(
        d_out_compact: &[f32],
        chunk_states: &[u16],
        d_qg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_state_matmul_body; d_out_compact, d_out_compact, chunk_states, d_qg, params; ChunkStateMatmulMode::Dqg);
    }

    #[kernel]
    pub fn chunk_kda_dkg_from_vnew_dh_kernel(
        v_new: &[f32],
        d_h_states: &[f32],
        d_kg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_kda_dkg_from_vnew_dh_body; v_new, d_h_states, d_kg, params);
    }

    #[kernel]
    pub fn chunkwise_kda_backward_kernel(
        qg: &[f32],
        kg: &[f32],
        u_to_du: DisjointSlice<f32>,
        w: &[f32],
        _aqk: &[f32],
        g: &[f32],
        chunk_states: &[u16],
        d_out: &[f32],
        d_h_states: DisjointSlice<f32>,
        _d_aqk: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        let inputs = KdaChunkwiseInputs {
            qg,
            kg,
            w,
            g,
            chunk_states,
            d_out,
        };
        let grads = KdaChunkwiseGrads {
            u_to_du,
            d_h_states,
        };
        with_kda_tiles!(backward chunkwise_kda_backward_body; inputs, grads, params);
    }

    #[kernel]
    pub fn chunk_intra_kda_dm_kernel(
        kg: &[f32],
        vbeta: &[f32],
        g: &[f32],
        beta: &[f32],
        d_u: &[f32],
        d_w: &[f32],
        d_m: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        with_tc_ab_tiles!(chunk_intra_kda_dm_body; KdaDmInputs { kg, vbeta, g, beta, d_u, d_w }, d_m, params);
    }

    #[kernel]
    pub fn chunk_intra_kda_backward_kernel(
        qg: &[f32],
        kg: &[f32],
        vbeta: &[f32],
        g: &[f32],
        beta: &[f32],
        d_qg_to_dv: DisjointSlice<f32>,
        d_kg: &[f32],
        d_k_a_to_dg: DisjointSlice<f32>,
        d_kpos_m: &[f32],
        d_vbeta_m: &[f32],
        d_kneg_from_b: &[f32],
        d_kpos_from_b_t: &[f32],
        d_q: DisjointSlice<f32>,
        d_k: DisjointSlice<f32>,
        d_beta: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        let inputs = KdaIntraInputs {
            qg,
            kg,
            vbeta,
            g,
            beta,
            d_kg,
            d_kpos_m,
            d_vbeta_m,
            d_kneg_from_b,
            d_kpos_from_b_t,
        };
        let grads = KdaIntraGrads {
            qg_to_dv: d_qg_to_dv,
            k_a_to_dg: d_k_a_to_dg,
            q: d_q,
            k: d_k,
            beta: d_beta,
        };
        chunk_intra_kda_backward_body(inputs, grads, params);
    }
}

pub(super) use module::{LoadedModule, from_module};
