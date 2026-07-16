use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::residual::ResidualBackwardModule;

use crate::backward::{
    BlockAttentionBackwardModules, BlockAttentionBackwardScratch, BlockAttentionBackwardSeeds,
    BlockMlpBackwardModules, FinalHeadBackwardModules, FinalHeadBackwardScratch,
    FinalHeadBackwardSeeds, MlpBackwardScratch, MlpBackwardSeeds,
};
use crate::types::{
    AttentionProjectionTensors, Gpt2BackwardGrads, Gpt2ForwardSaved, LayerNormTensors,
    MlpProjectionTensors,
};
use crate::{GPT2_N_LAYER, Gpt2Rng};

#[derive(Clone, Copy)]
pub struct Gpt2BackwardModules<'a> {
    pub residual: &'a ResidualBackwardModule,
    pub final_head: FinalHeadBackwardModules<'a>,
    pub final_norm: &'a rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule,
    pub attention: BlockAttentionBackwardModules<'a>,
    pub mlp: BlockMlpBackwardModules<'a>,
}

#[derive(Clone, Copy)]
pub struct Gpt2BackwardWeights<'a> {
    pub lm_head_weight: Nvfp4DeviceTensor<'a>,
    pub ln_f: LayerNormTensors<'a>,
    pub block_ln_1: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub block_ln_2: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub attention: [AttentionProjectionTensors<'a>; GPT2_N_LAYER],
    pub mlp: [MlpProjectionTensors<'a>; GPT2_N_LAYER],
}

pub struct Gpt2BackwardScratch<'scratch> {
    pub final_head: FinalHeadBackwardScratch<'scratch>,
    pub attention: BlockAttentionBackwardScratch<'scratch>,
    pub mlp: MlpBackwardScratch<'scratch>,
    pub d_residual_after_attention: &'scratch mut DeviceBuffer<f32>,
    pub d_hidden: &'scratch mut DeviceBuffer<f32>,
    pub d_qkv: &'scratch mut DeviceBuffer<f32>,
    pub d_mlp_up: &'scratch mut DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub struct Gpt2BackwardSeeds {
    pub final_head: FinalHeadBackwardSeeds,
    pub attention: [BlockAttentionBackwardSeeds; GPT2_N_LAYER],
    pub mlp: [MlpBackwardSeeds; GPT2_N_LAYER],
}

impl Gpt2BackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            final_head: FinalHeadBackwardSeeds::from_rng(rng),
            attention: std::array::from_fn(|_| BlockAttentionBackwardSeeds::from_rng(rng)),
            mlp: std::array::from_fn(|_| MlpBackwardSeeds::from_rng(rng)),
        }
    }
}

pub struct Gpt2BackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: Gpt2BackwardModules<'a>,
    pub saved: Gpt2ForwardSaved<'a>,
    pub weights: Gpt2BackwardWeights<'a>,
    pub targets: &'a DeviceBuffer<u32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub extra_final_normalized_grad: Option<&'a DeviceBuffer<f32>>,
    pub d_lm_head_weight: &'out mut DeviceBuffer<f32>,
    pub grads: Gpt2BackwardGrads<'out>,
    pub scratch: Gpt2BackwardScratch<'scratch>,
    pub seeds: Gpt2BackwardSeeds,
}
