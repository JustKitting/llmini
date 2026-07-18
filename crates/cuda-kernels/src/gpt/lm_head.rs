use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError};

use crate::launch::launch_config;
use crate::mma::{
    NVFP4_PROJECTION_CTA_THREADS, Nvfp4FourSixMmaWeightTensor, projection_cta_launch_grid_dim,
};
use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use crate::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

#[path = "lm_head/kernels.rs"]
mod kernels;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LmHeadParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
    pub weight_global_scale: f32,
}

unsafe impl DeviceCopy for LmHeadParams {}

pub struct LmHeadArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4FourSixMmaWeightTensor<'a>,
    pub logits: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
}

pub struct LmHeadTmaArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub tma: &'a Nvfp4GemmModule,
    pub scale_pack: &'a Sm120ScalePackModule,
    pub descriptors: &'scratch mut TmaNvfp4DeviceScaleDescriptors,
    pub input_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub weight: Nvfp4DeviceTensor<'a>,
    pub weight_scale_packed: &'scratch mut DeviceBuffer<u8>,
    pub logits: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub input_dim: u32,
    pub vocab_size: u32,
}

pub struct LmHeadModule {
    module: kernels::LoadedModule,
}

impl LmHeadModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module)?,
        })
    }

    pub fn logits(&self, args: LmHeadArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.lm_head_kernel(
            args.stream,
            launch_config(
                projection_cta_launch_grid_dim(args.token_count, args.input_dim, args.vocab_size),
                NVFP4_PROJECTION_CTA_THREADS,
            ),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.weight.bytes,
            args.weight.scales,
            args.weight.global_scale,
            args.logits,
            LmHeadParams {
                token_count: args.token_count,
                input_dim: args.input_dim,
                vocab_size: args.vocab_size,
                weight_global_scale: 1.0,
            },
        )
    }

    pub fn logits_tma(&self, args: LmHeadTmaArgs<'_, '_, '_>) -> Result<(), DriverError> {
        args.scale_pack.pack(
            args.stream,
            args.input.scales,
            args.input_scale_packed,
            args.token_count,
            args.input_dim,
        )?;
        args.scale_pack.pack(
            args.stream,
            args.weight.scales,
            args.weight_scale_packed,
            args.vocab_size,
            args.input_dim,
        )?;
        args.tma.prepare_tma_nvfp4_device_scales_into(
            args.stream,
            args.input.bytes,
            args.input_scale_packed,
            args.weight.bytes,
            args.weight_scale_packed,
            args.token_count,
            args.input_dim,
            args.vocab_size,
            args.descriptors,
        )?;
        args.tma
            .gemm_tma_nvfp4_rowwise_a_scale_and_global_scale_buffer(
                args.stream,
                args.descriptors,
                args.logits,
                args.token_count,
                args.input_dim,
                args.vocab_size,
                args.input.global_scales,
                args.weight.global_scale,
                None,
            )
    }
}
