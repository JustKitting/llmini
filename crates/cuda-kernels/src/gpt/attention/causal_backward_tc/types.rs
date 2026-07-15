use cuda_core::{CudaStream, DeviceBuffer};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::F16TcMatmulModule;

pub struct CausalAttentionBackwardTcScratch<'a> {
    pub q_f32: &'a mut DeviceBuffer<f32>,
    pub k_f32: &'a mut DeviceBuffer<f32>,
    pub v_f32: &'a mut DeviceBuffer<f32>,
    pub g_f32: &'a mut DeviceBuffer<f32>,
    pub q: &'a mut DeviceBuffer<u16>,
    pub k: &'a mut DeviceBuffer<u16>,
    pub v: &'a mut DeviceBuffer<u16>,
    pub d_out: &'a mut DeviceBuffer<u16>,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub dot: &'a mut DeviceBuffer<f32>,
    pub p: &'a mut DeviceBuffer<f32>,
    pub ds: &'a mut DeviceBuffer<f32>,
    pub p_half: &'a mut DeviceBuffer<u16>,
    pub ds_half: &'a mut DeviceBuffer<u16>,
    pub d_q: &'a mut DeviceBuffer<f32>,
    pub d_k: &'a mut DeviceBuffer<f32>,
    pub d_v: &'a mut DeviceBuffer<f32>,
    pub kda_d_q: &'a mut DeviceBuffer<f32>,
    pub kda_d_k: &'a mut DeviceBuffer<f32>,
    pub kda_d_v: &'a mut DeviceBuffer<f32>,
    pub kda_d_g: &'a mut DeviceBuffer<f32>,
    pub kda_d_beta: &'a mut DeviceBuffer<f32>,
}

pub struct CausalAttentionBackwardTcArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub tc_module: &'a F16TcMatmulModule,
    pub qkv: &'a DeviceBuffer<u16>,
    pub attention_out: &'a DeviceBuffer<u16>,
    pub kda_v_new: Option<&'a DeviceBuffer<f32>>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub log_sum_exp: &'a DeviceBuffer<f32>,
    pub softmax_d: &'scratch mut DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub scratch: CausalAttentionBackwardTcScratch<'scratch>,
    pub row_count: u32,
    pub seq_len: u32,
    pub batch_size: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
}

impl CausalAttentionBackwardTcArgs<'_, '_, '_> {
    pub(super) fn params(&self) -> CausalAttentionParams {
        CausalAttentionParams::new(
            self.row_count,
            self.seq_len,
            self.batch_size,
            self.embedding_dim,
            self.qkv_dim,
            self.head_count,
            self.head_dim,
        )
    }
}

impl<'a> CausalAttentionBackwardTcScratch<'a> {
    pub fn reborrow(&mut self) -> CausalAttentionBackwardTcScratch<'_> {
        CausalAttentionBackwardTcScratch {
            q_f32: self.q_f32,
            k_f32: self.k_f32,
            v_f32: self.v_f32,
            g_f32: self.g_f32,
            q: self.q,
            k: self.k,
            v: self.v,
            d_out: self.d_out,
            scores: self.scores,
            dot: self.dot,
            p: self.p,
            ds: self.ds,
            p_half: self.p_half,
            ds_half: self.ds_half,
            d_q: self.d_q,
            d_k: self.d_k,
            d_v: self.d_v,
            kda_d_q: self.kda_d_q,
            kda_d_k: self.kda_d_k,
            kda_d_v: self.kda_d_v,
            kda_d_g: self.kda_d_g,
            kda_d_beta: self.kda_d_beta,
        }
    }
}
