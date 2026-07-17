use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use crate::launch::{grid_x_config, launch_config};
use crate::nvfp4::Nvfp4DeviceTensor;

use super::{GPT_LAYER_NORM_THREADS_PER_BLOCK, THREADS_PER_BLOCK, WARPS_PER_BLOCK, kernels};

pub struct LayerNormArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub gamma: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub epsilon: f32,
}

macro_rules! gpt_layer_norm_args {
    ($name:ident $(, $extra:ident: $extra_ty:ty)*) => {
        pub struct $name<'a, 'out> {
            pub stream: &'a CudaStream,
            pub residual: &'a DeviceBuffer<f32>,
            pub weight: Nvfp4DeviceTensor<'a>,
            pub bias: Nvfp4DeviceTensor<'a>,
            pub normalized: &'out mut DeviceBuffer<f32>,
            pub normalized_amax: &'out mut DeviceBuffer<f32>,
            pub mean: &'out mut DeviceBuffer<f32>,
            pub inv_std: &'out mut DeviceBuffer<f32>,
            $(pub $extra: $extra_ty,)*
            pub row_count: u32,
            pub embedding_dim: u32,
            pub epsilon: f32,
            pub output_scale: f32,
        }
    };
}

gpt_layer_norm_args!(GptLayerNormArgs);
gpt_layer_norm_args!(GptLayerNormSaveResidualF16Args, residual_f16: &'out mut DeviceBuffer<u16>);

pub struct LayerNormModule {
    module: kernels::LoadedModule,
}

macro_rules! gpt_layer_norm_launcher {
    ($method:ident, $args:ty, $kernel:ident $(, $extra:ident)*) => {
        pub fn $method(&self, args: $args) -> Result<(), DriverError> {
            self.module.$kernel(
                args.stream,
                grid_x_config(args.row_count, GPT_LAYER_NORM_THREADS_PER_BLOCK),
                args.residual, args.weight.bytes, args.weight.scales, args.bias.bytes,
                args.bias.scales, args.weight.global_scale, args.bias.global_scale,
                args.normalized, args.normalized_amax, args.mean, args.inv_std, $(args.$extra,)*
                args.row_count, args.embedding_dim, args.epsilon, args.output_scale,
            )
        }
    };
}

impl LayerNormModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn layer_norm_warp_f32(&self, args: LayerNormArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.layer_norm_warp_f32_kernel(
            args.stream,
            launch_config(
                (args.row_count.div_ceil(WARPS_PER_BLOCK), 1, 1),
                THREADS_PER_BLOCK,
            ),
            args.x,
            args.gamma,
            args.beta,
            args.out,
            args.row_count,
            args.epsilon,
        )
    }

    gpt_layer_norm_launcher!(
        gpt_layer_norm,
        GptLayerNormArgs<'_, '_>,
        gpt_layer_norm_kernel
    );
    gpt_layer_norm_launcher!(
        gpt_layer_norm_save_residual_f16,
        GptLayerNormSaveResidualF16Args<'_, '_>,
        gpt_layer_norm_save_residual_f16_kernel,
        residual_f16
    );
}
