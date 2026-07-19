use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionTcScratch};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::tape::AttentionForwardTape;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4};

#[derive(Clone, Copy)]
pub struct AttentionProjectionTensors<'a> {
    pub qkv_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub qkv_weight_device: Nvfp4DeviceTensor<'a>,
    pub qkv_bias: Nvfp4DeviceTensor<'a>,
    pub qk_scale: Nvfp4DeviceTensor<'a>,
    pub xsa_alphas: Nvfp4DeviceTensor<'a>,
    pub xsa_alpha_offset: u32,
    pub c_proj_weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub c_proj_weight_device: Nvfp4DeviceTensor<'a>,
    pub c_proj_bias: Nvfp4DeviceTensor<'a>,
}

pub struct AttentionForwardArgs<'a, 'scratch> {
    pub block_index: usize,
    pub use_full_attention: bool,
    pub module: &'a AttentionModule,
    pub tc_module: &'a F16TcMatmulModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub tma_scale_pack: &'a Sm120ScalePackModule,
    pub tma_pad: &'a TmaMatrixPadModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub tc_scratch: CausalAttentionTcScratch<'scratch>,
    pub tma_descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
    pub tma_input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_weight_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_weight_bytes_padded: &'scratch mut DeviceBuffer<u8>,
    pub projections: AttentionProjectionTensors<'a>,
    pub qkv: &'scratch mut DeviceBuffer<f32>,
    pub value_residual: &'scratch mut DeviceBuffer<f32>,
    pub attention_log_sum_exp: &'scratch mut DeviceBuffer<f32>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<AttentionForwardTape<'scratch>>,
}
