use cuda_core::DeviceBuffer;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::tape::MlpForwardTape;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4, MlpActivationNvfp4};

#[derive(Clone, Copy)]
pub struct MlpUpTensors<'a> {
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub weight_device: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct MlpDownTensors<'a> {
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub weight_device: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct MlpProjectionTensors<'a> {
    pub up: MlpUpTensors<'a>,
    pub down: MlpDownTensors<'a>,
}

pub struct MlpScratch<'scratch> {
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub pre_activation: &'scratch mut DeviceBuffer<f32>,
    pub activation: &'scratch mut DeviceBuffer<f32>,
    pub tma_descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
    pub tma_input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_wide_input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub tma_weight_scale_packed: &'scratch mut DeviceBuffer<u8>,
}

pub struct MlpForwardArgs<'a, 'scratch> {
    pub module: &'a MlpModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub tma_scale_pack: &'a Sm120ScalePackModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub scratch: MlpScratch<'scratch>,
    pub projections: MlpProjectionTensors<'a>,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<MlpForwardTape<'scratch>>,
}
