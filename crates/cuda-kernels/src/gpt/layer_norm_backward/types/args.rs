use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::Nvfp4DeviceTensor;

macro_rules! input_args {
    ($name:ident, $residual:ty) => {
        pub struct $name<'a, 'out> {
            pub stream: &'a CudaStream,
            pub residual: &'a DeviceBuffer<$residual>,
            pub d_normalized: &'a DeviceBuffer<f32>,
            pub mean: &'a DeviceBuffer<f32>,
            pub inv_std: &'a DeviceBuffer<f32>,
            pub weight: Nvfp4DeviceTensor<'a>,
            pub d_residual: &'out mut DeviceBuffer<f32>,
            pub row_count: u32,
            pub embedding_dim: u32,
        }
    };
}

macro_rules! param_args {
    ($name:ident, $residual:ty) => {
        pub struct $name<'a, 'out> {
            pub stream: &'a CudaStream,
            pub residual: &'a DeviceBuffer<$residual>,
            pub d_normalized: &'a DeviceBuffer<f32>,
            pub mean: &'a DeviceBuffer<f32>,
            pub inv_std: &'a DeviceBuffer<f32>,
            pub d_weight: &'out mut DeviceBuffer<f32>,
            pub d_bias: &'out mut DeviceBuffer<f32>,
            pub row_count: u32,
            pub embedding_dim: u32,
        }
    };
}

input_args!(LayerNormBackwardInputArgs, u16);
input_args!(LayerNormBackwardInputF32Args, f32);

pub struct LayerNormBackwardInputAddArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub residual: &'a DeviceBuffer<u16>,
    pub d_normalized: &'a DeviceBuffer<f32>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub direct: &'a DeviceBuffer<f32>,
    pub d_residual: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub embedding_dim: u32,
}

param_args!(LayerNormBackwardParamArgs, u16);
param_args!(LayerNormBackwardParamF32Args, f32);
