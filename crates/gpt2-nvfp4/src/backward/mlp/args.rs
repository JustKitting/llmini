use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::linear_backward::{LinearBackwardModule, LinearBackwardMsEdenScratch};
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::Gpt2Rng;
use crate::backward::scratch_reborrow::reborrow_ms_eden;
use crate::types::{BlockForwardSaved, MlpProjectionTensors};

#[derive(Clone, Copy)]
pub struct MlpBackwardModules<'a> {
    pub transpose: &'a TransposeModule,
    pub decode: &'a Nvfp4DecodeModule,
    pub linear: &'a LinearBackwardModule,
    pub quant: &'a Nvfp4QuantModule,
    pub mlp: &'a MlpModule,
}

pub struct MlpBackwardScratch<'scratch> {
    pub down_linear: LinearBackwardMsEdenScratch<'scratch>,
    pub up_linear: LinearBackwardMsEdenScratch<'scratch>,
}

impl<'scratch> MlpBackwardScratch<'scratch> {
    pub fn reborrow(&mut self) -> MlpBackwardScratch<'_> {
        MlpBackwardScratch {
            down_linear: reborrow_ms_eden(&mut self.down_linear),
            up_linear: reborrow_ms_eden(&mut self.up_linear),
        }
    }
}

pub struct MlpBackwardGrads<'out> {
    pub d_mlp_up: &'out mut DeviceBuffer<f32>,
    pub d_ln_2_normalized: &'out mut DeviceBuffer<f32>,
    pub d_c_proj_weight: &'out mut DeviceBuffer<f32>,
    pub d_c_proj_bias: &'out mut DeviceBuffer<f32>,
    pub d_c_fc_weight: &'out mut DeviceBuffer<f32>,
    pub d_c_fc_bias: &'out mut DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub struct MlpBackwardSeeds {
    pub(crate) down_sign: u32,
    pub(crate) down_scale: u32,
    pub(crate) up_sign: u32,
    pub(crate) up_scale: u32,
}

impl MlpBackwardSeeds {
    pub fn from_rng(rng: &mut Gpt2Rng) -> Self {
        Self {
            down_sign: rng.next_u32(),
            down_scale: rng.next_u32(),
            up_sign: rng.next_u32(),
            up_scale: rng.next_u32(),
        }
    }
}

pub struct MlpBackwardArgs<'a, 'scratch, 'out> {
    pub block_index: usize,
    pub stream: &'a CudaStream,
    pub modules: MlpBackwardModules<'a>,
    pub saved: BlockForwardSaved<'a>,
    pub projections: MlpProjectionTensors<'a>,
    pub d_residual_out: &'a DeviceBuffer<f32>,
    pub precomputed_d_residual_amax_chunks: Option<u32>,
    pub grads: MlpBackwardGrads<'out>,
    pub scratch: MlpBackwardScratch<'scratch>,
    pub seeds: MlpBackwardSeeds,
}
