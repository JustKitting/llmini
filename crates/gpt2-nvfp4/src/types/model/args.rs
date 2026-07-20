use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::canon::CanonModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use crate::GPT2_N_LAYER;
use crate::types::{
    AttentionProjectionTensors, CanonTensors, Gpt2ForwardTape, HiddenStateNvfp4, LayerNormTensors,
    MlpActivationNvfp4, MlpProjectionTensors, TokenEmbeddingArgs,
};

pub struct Gpt2ForwardArgs<'a> {
    pub embeddings: TokenEmbeddingArgs<'a>,
    pub attention_module: &'a AttentionModule,
    pub canon_module: &'a CanonModule,
    pub attention_tc_module: &'a F16TcMatmulModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub lm_head_module: &'a LmHeadModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub tma_scale_pack: &'a Sm120ScalePackModule,
    pub tma_pad: &'a TmaMatrixPadModule,
    pub tma_descriptors: &'a mut TmaNvfp4DeviceScaleDescriptors,
    pub tma_input_scale_packed: &'a mut DeviceBuffer<u8>,
    pub tma_wide_input_scale_packed: &'a mut DeviceBuffer<u8>,
    pub tma_weight_scale_packed: &'a mut DeviceBuffer<u8>,
    pub tma_weight_bytes_padded: &'a mut DeviceBuffer<u8>,
    pub hidden_nvfp4: HiddenStateNvfp4<'a>,
    pub attention_tc_scratch: CausalAttentionTcScratch<'a>,
    pub mlp_activation_nvfp4: MlpActivationNvfp4<'a>,
    pub attention: [AttentionProjectionTensors<'a>; GPT2_N_LAYER],
    pub block_ln_1: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub block_canon_a: [CanonTensors<'a>; GPT2_N_LAYER],
    pub block_ln_2: [LayerNormTensors<'a>; GPT2_N_LAYER],
    pub block_canon_c: [CanonTensors<'a>; GPT2_N_LAYER],
    pub mlp: [MlpProjectionTensors<'a>; GPT2_N_LAYER],
    pub ln_f: LayerNormTensors<'a>,
    pub attention_qkv: &'a mut DeviceBuffer<f32>,
    pub attention_value_residual: &'a mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'a mut DeviceBuffer<f32>,
    pub mlp_activation: &'a mut DeviceBuffer<f32>,
    pub mlp_route_scores: &'a mut DeviceBuffer<f32>,
    pub mlp_route_masks: &'a mut DeviceBuffer<u64>,
    pub logits: &'a mut DeviceBuffer<f32>,
    pub tape: Option<Gpt2ForwardTape<'a>>,
}
