use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use crate::nvfp4_quant::Nvfp4QuantModule;

use super::scratch::LinearBackwardMsEdenScratch;

#[derive(Clone, Copy)]
pub enum LinearBackwardInputTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    RowwiseNvfp4(Nvfp4RowwiseDeviceTensor<'a>),
}

#[derive(Clone, Copy)]
pub enum LinearBackwardWeightTranspose<'a> {
    Fp32(&'a DeviceBuffer<f32>),
    Nvfp4(Nvfp4DeviceTensor<'a>),
}

pub struct LinearBackwardMsEdenArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub weight_t: LinearBackwardWeightTranspose<'a>,
    pub input_t: LinearBackwardInputTranspose<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dinput: &'out mut DeviceBuffer<f32>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub dbias: Option<&'out mut DeviceBuffer<f32>>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub precomputed_e_amax_chunks: Option<u32>,
}

pub struct LinearBackwardMsEdenWeightArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub quant_module: &'a Nvfp4QuantModule,
    pub e: &'a DeviceBuffer<f32>,
    pub input_t: LinearBackwardInputTranspose<'a>,
    pub scratch: LinearBackwardMsEdenScratch<'scratch>,
    pub dweight: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}
