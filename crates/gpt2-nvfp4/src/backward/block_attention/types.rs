use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;

use crate::Gpt2Rng;
use crate::LayerNormTensors;
use crate::backward::{
    AttentionBackwardModules, AttentionBackwardSeeds, AttentionCProjScratch, AttentionCoreScratch,
    AttentionQkvScratch,
};
use crate::types::{AttentionProjectionTensors, BlockBackwardGrads, BlockForwardSaved};

#[derive(Clone, Copy)]
pub struct BlockAttentionBackwardModules<'a> {
    pub residual: &'a ResidualBackwardModule,
    pub layer_norm: &'a LayerNormBackwardModule,
    pub attention: &'a AttentionModule,
    pub f16_tc: &'a F16TcMatmulModule,
    pub linear: AttentionBackwardModules<'a>,
}

pub struct BlockAttentionBackwardScratch<'scratch> {
    pub c_proj: AttentionCProjScratch<'scratch>,
    pub core: AttentionCoreScratch<'scratch>,
    pub qkv: AttentionQkvScratch<'scratch>,
}

impl<'scratch> BlockAttentionBackwardScratch<'scratch> {
    pub fn reborrow(&mut self) -> BlockAttentionBackwardScratch<'_> {
        BlockAttentionBackwardScratch {
            c_proj: self.c_proj.reborrow(),
            core: self.core.reborrow(),
            qkv: self.qkv.reborrow(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct BlockAttentionBackwardSeeds {
    pub c_proj: AttentionBackwardSeeds,
    pub qkv: AttentionBackwardSeeds,
}

impl BlockAttentionBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            c_proj: AttentionBackwardSeeds::from_rng(rng),
            qkv: AttentionBackwardSeeds::from_rng(rng),
        }
    }
}

pub struct BlockAttentionBackwardArgs<'a, 'scratch, 'out> {
    pub block_index: usize,
    pub use_full_attention: bool,
    pub reuse_forward_probs: bool,
    pub stream: &'a CudaStream,
    pub modules: BlockAttentionBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub ln_1: LayerNormTensors<'a>,
    pub projections: AttentionProjectionTensors<'a>,
    pub d_residual_after_attention: &'scratch DeviceBuffer<f32>,
    pub d_residual_in: &'scratch mut DeviceBuffer<f32>,
    pub d_hidden: &'scratch mut DeviceBuffer<f32>,
    pub d_qkv: &'scratch mut DeviceBuffer<f32>,
    pub grads: BlockBackwardGrads<'out>,
    pub scratch: BlockAttentionBackwardScratch<'scratch>,
    pub seeds: BlockAttentionBackwardSeeds,
}
