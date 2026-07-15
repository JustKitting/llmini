use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::linear_backward::{LinearBackwardModule, LinearBackwardMsEdenScratch};
use rust_kernels_cuda::loss::LossModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::Gpt2Rng;

#[derive(Clone, Copy)]
pub struct FinalHeadBackwardModules<'a> {
    pub loss: &'a LossModule,
    pub transpose: &'a TransposeModule,
    pub decode: &'a Nvfp4DecodeModule,
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
}

pub struct FinalHeadBackwardScratch<'scratch> {
    pub linear: LinearBackwardMsEdenScratch<'scratch>,
}

#[derive(Clone, Copy)]
pub struct FinalHeadBackwardSeeds {
    pub(crate) sign: u32,
    pub(crate) scale: u32,
}

impl FinalHeadBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            sign: rng.next_u32(),
            scale: rng.next_u32(),
        }
    }
}

pub struct FinalHeadBackwardArgs<'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: FinalHeadBackwardModules<'a>,
    pub logits: &'a DeviceBuffer<f32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub final_normalized: Nvfp4RowwiseDeviceTensor<'a>,
    pub lm_head_weight: Nvfp4DeviceTensor<'a>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub dlogits: &'out mut DeviceBuffer<f32>,
    pub d_final_normalized: &'out mut DeviceBuffer<f32>,
    pub d_lm_head_weight: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub scratch: FinalHeadBackwardScratch<'scratch>,
    pub seeds: FinalHeadBackwardSeeds,
}
