use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

use super::{
    AttentionBackwardModules, AttentionBackwardSeeds, AttentionCProjScratch, AttentionCoreScratch,
    AttentionQkvScratch,
};
use crate::types::{AttentionProjectionTensors, BlockForwardSaved};

pub struct AttentionCProjBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: AttentionBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub projections: AttentionProjectionTensors<'a>,
    pub d_residual_after_attention: &'a DeviceBuffer<f32>,
    pub precomputed_d_residual_amax_chunks: Option<u32>,
    pub d_attention_out: &'out mut DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: &'out mut DeviceBuffer<f32>,
    pub d_attn_c_proj_bias: &'out mut DeviceBuffer<f32>,
    pub scratch: AttentionCProjScratch<'scratch>,
    pub seeds: AttentionBackwardSeeds,
}

pub struct AttentionCoreBackwardArgs<'a, 'scratch, 'out> {
    pub block_index: usize,
    pub use_full_attention: bool,
    pub reuse_forward_probs: bool,
    pub stream: &'a CudaStream,
    pub module: &'a AttentionModule,
    pub tc_module: &'a F16TcMatmulModule,
    pub saved: BlockForwardSaved<'a>,
    pub d_attention_out: &'a DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_qkv_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub scratch: AttentionCoreScratch<'scratch>,
    pub backward_mask_seed: u32,
}

pub struct AttentionQkvBackwardArgs<'a, 'scratch, 'out> {
    pub use_full_attention: bool,
    pub stream: &'a CudaStream,
    pub modules: AttentionBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub projections: AttentionProjectionTensors<'a>,
    pub d_qkv: &'a DeviceBuffer<f32>,
    pub d_ln_1_normalized: &'out mut DeviceBuffer<f32>,
    pub d_attn_qkv_weight: &'out mut DeviceBuffer<f32>,
    pub d_attn_qkv_bias: &'out mut DeviceBuffer<f32>,
    pub precomputed_d_qkv_amax_chunks: Option<u32>,
    pub scratch: AttentionQkvScratch<'scratch>,
    pub seeds: AttentionBackwardSeeds,
}
