use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use crate::types::{
    AttentionProjectionTensors, BlockForwardTape, HiddenStateDevice, HiddenStateNvfp4,
    LayerNormTensors, MlpActivationNvfp4, MlpProjectionTensors,
};

pub struct BlockForwardArgs<'a, 'scratch> {
    pub use_full_attention: bool,
    pub attention_module: &'a AttentionModule,
    pub attention_tc_module: &'a F16TcMatmulModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub layer_norm_module: &'a LayerNormModule,
    pub mlp_module: &'a MlpModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub tma_scale_pack: &'a Sm120ScalePackModule,
    pub tma_pad: &'a TmaMatrixPadModule,
    pub hidden_nvfp4: HiddenStateNvfp4<'scratch>,
    pub attention_tc_scratch: CausalAttentionTcScratch<'scratch>,
    pub mlp_activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub tma_descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
    pub tma_input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_wide_input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_weight_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_weight_bytes_padded: &'scratch mut DeviceBuffer<u8>,
    pub projections: AttentionProjectionTensors<'a>,
    pub ln_1: LayerNormTensors<'a>,
    pub ln_2: LayerNormTensors<'a>,
    pub mlp: MlpProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub mlp_pre_activation: &'scratch mut DeviceBuffer<f32>,
    pub mlp_activation: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<BlockForwardTape<'scratch>>,
}
