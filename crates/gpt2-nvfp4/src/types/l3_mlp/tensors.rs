use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmModule;
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;

use super::tape::MlpForwardTape;
use crate::types::{HiddenStateDevice, HiddenStateNvfp4, MlpActivationNvfp4};

/// Optional correctness observer; production training supplies None.
pub trait MlpProjectionObserver {
    fn observe(
        &self,
        block: usize,
        stream: &CudaStream,
        paired: &DeviceBuffer<f32>,
        elements: usize,
        recomputed: bool,
    ) -> Result<(), DriverError>;
}

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
    pub paired: &'scratch mut DeviceBuffer<f32>,
    pub input_nvfp4: HiddenStateNvfp4<'scratch>,
    pub activation_nvfp4: MlpActivationNvfp4<'scratch>,
    pub activation: &'scratch mut DeviceBuffer<f32>,
    pub route_scores: &'scratch mut DeviceBuffer<f32>,
    pub route_masks: &'scratch mut DeviceBuffer<u64>,
    pub tma_descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
}

pub struct MlpForwardArgs<'a, 'scratch> {
    pub projection_observer: Option<&'a dyn MlpProjectionObserver>,
    pub block_index: usize,
    pub module: &'a MlpModule,
    pub tma_module: &'a Nvfp4GemmModule,
    pub quant_module: &'a Nvfp4QuantModule,
    pub scratch: MlpScratch<'scratch>,
    pub projections: MlpProjectionTensors<'a>,
    pub input_prequantized: bool,
    pub hidden: HiddenStateDevice<'a>,
    pub tape: Option<MlpForwardTape<'scratch>>,
}
