use cuda_core::{CudaStream, DeviceBuffer};

use crate::mma::Nvfp4FourSixMmaWeightTensor;
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

pub struct MlpUpRelu2Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub pre_activation: &'out mut DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct MlpDownResidualArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub bias: Nvfp4DeviceTensor<'a>,
    pub residual: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct Relu2BackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub pre_activation: &'a DeviceBuffer<f32>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub d_pre_activation: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}

pub struct Relu2BackwardF16Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub pre_activation: &'a DeviceBuffer<u16>,
    pub d_out: &'a DeviceBuffer<f32>,
    pub d_pre_activation: &'out mut DeviceBuffer<f32>,
    pub d_pre_activation_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub len: u32,
}
