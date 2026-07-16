use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{
    Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, Nvfp4QuantRowwiseDerivedAmaxArgs,
};

use crate::types::HiddenStateDevice;

pub struct RowwiseNvfp4Scratch<'a> {
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scales: &'a mut DeviceBuffer<f32>,
}

impl<'a> RowwiseNvfp4Scratch<'a> {
    pub fn quantize_precomputed_amax(
        &mut self,
        quant_module: &Nvfp4QuantModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        amax: &DeviceBuffer<f32>,
        row_count: u32,
        row_len: u32,
    ) -> Result<(), DriverError> {
        quant_module.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x: input,
            amax,
            out_fp4: self.bytes,
            out_scales: self.scales,
            out_global_scale: self.global_scales,
            group_count: row_count * row_len / 16,
            row_len,
        })
    }

    pub fn quantize_row_amax(
        &mut self,
        quant_module: &Nvfp4QuantModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        amax: &mut DeviceBuffer<f32>,
        row_count: u32,
        row_len: u32,
    ) -> Result<(), DriverError> {
        quant_module.fp32_to_nvfp4_four_six_rowwise_derived_amax(Nvfp4QuantRowwiseDerivedAmaxArgs {
            stream,
            x: input,
            amax,
            out_fp4: self.bytes,
            out_scales: self.scales,
            out_global_scale: self.global_scales,
            row_count,
            row_len,
        })
    }

    pub fn quantize_hidden_precomputed(
        &mut self,
        quant_module: &Nvfp4QuantModule,
        hidden: &HiddenStateDevice<'_>,
        row_len: u32,
    ) -> Result<Nvfp4RowwiseDeviceTensor<'_>, DriverError> {
        self.quantize_precomputed_amax(
            quant_module,
            hidden.stream,
            &*hidden.normalized,
            &*hidden.normalized_amax,
            hidden.row_count,
            row_len,
        )?;
        Ok(self.device())
    }

    pub fn device(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&*self.bytes, &*self.scales, &*self.global_scales)
    }

    pub fn reborrow(&mut self) -> RowwiseNvfp4Scratch<'_> {
        RowwiseNvfp4Scratch {
            bytes: &mut *self.bytes,
            scales: &mut *self.scales,
            global_scales: &mut *self.global_scales,
        }
    }
}

pub type HiddenStateNvfp4<'a> = RowwiseNvfp4Scratch<'a>;
pub type MlpActivationNvfp4<'a> = RowwiseNvfp4Scratch<'a>;
