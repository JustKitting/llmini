mod adam;
mod embedding;
mod ember;
mod grad_clip;
mod kda_clip;
mod muon_mega;
mod symexp_lin;
mod update;

use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use super::modules;
use crate::nvfp4_quant::{Nvfp4QuantArgs, Nvfp4QuantModule, TensorAmaxArgs};
use crate::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule, scale_pack::Sm120ScalePackModule,
};

pub struct OptimizerModule {
    pub(super) apply: modules::LoadedModule,
    quant: Nvfp4QuantModule,
    pub(super) tma: Nvfp4GemmModule,
    pub(super) tma_scale_pack: Sm120ScalePackModule,
    pub(super) tma_pad: TmaMatrixPadModule,
}

impl OptimizerModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            apply: modules::from_module(module.clone())?,
            quant: Nvfp4QuantModule::from_module(module.clone())?,
            tma: Nvfp4GemmModule::from_module(module.clone())?,
            tma_scale_pack: Sm120ScalePackModule::from_module(module.clone())?,
            tma_pad: TmaMatrixPadModule::from_module(module)?,
        })
    }

    pub fn tma_gemm(&self) -> &Nvfp4GemmModule {
        &self.tma
    }

    pub fn tma_scale_pack(&self) -> &Sm120ScalePackModule {
        &self.tma_scale_pack
    }

    pub fn tma_pad(&self) -> &TmaMatrixPadModule {
        &self.tma_pad
    }

    #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
    pub fn materialize_master(
        &self,
        stream: &CudaStream,
        bytes: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
        global_scale: &mut DeviceBuffer<f32>,
        master: &DeviceBuffer<f32>,
        amax: &mut DeviceBuffer<f32>,
        chunk_amax: &mut DeviceBuffer<f32>,
        len: u32,
    ) -> Result<(), DriverError> {
        self.quant.tensor_amax_f32(TensorAmaxArgs {
            stream,
            x: master,
            chunk_amax,
            out: amax,
            element_count: len,
        })?;

        self.quant.fp32_to_nvfp4_four_six(Nvfp4QuantArgs {
            stream,
            x: master,
            amax: &*amax,
            out_fp4: bytes,
            out_scales: scales,
            out_global_scale: global_scale,
            group_count: len / 16,
        })
    }
}
