use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::types::HiddenStateDevice;

#[derive(Clone, Copy)]
pub struct LayerNormTensors<'a> {
    pub weight: Nvfp4DeviceTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
}

pub struct LayerNormForwardArgs<'a> {
    pub module: &'a LayerNormModule,
    pub tensors: LayerNormTensors<'a>,
    pub hidden: HiddenStateDevice<'a>,
    pub output_scale: f32,
}
