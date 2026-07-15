use cuda_core::{DeviceBuffer, DriverError};

use crate::linear_backward::LinearBackwardMsEdenScratch;
use crate::nvfp4_quant::MsEdenPairDeviceScaleQuantArgs;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

use super::context::QuantizeContext;

impl<'a> QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden) fn error_pair(
        &self,
        e: &DeviceBuffer<f32>,
        scratch: &mut LinearBackwardMsEdenScratch<'_>,
        precomputed_chunk_count: Option<u32>,
        dbias: Option<&mut DeviceBuffer<f32>>,
    ) -> Result<bool, DriverError> {
        let args = MsEdenPairDeviceScaleQuantArgs {
            stream: self.stream,
            x: e,
            out_fp4: &mut *scratch.e_h.bytes,
            out_scales: &mut *scratch.e_h.scales,
            out_global_scales: &mut *scratch.e_h.global_scales,
            transpose_out_fp4: &mut *scratch.e_t_h.bytes,
            transpose_out_scales: &mut *scratch.e_t_h.scales,
            transpose_out_global_scales: &mut *scratch.e_t_h.global_scales,
            out_chunk_amax: &mut *scratch.e_h.chunk_amax,
            out_global_scale: &mut *scratch.e_h.global_scale,
            row_count: self.token_count,
            src_row_len: self.output_dim,
            dst_row_len: self.output_k,
            transpose_dst_row_len: self.token_k,
            scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
            sign_seed: self.sign_seed,
            scale_seed: self.scale_seed,
            transpose_scale_seed: self.scale_seed ^ 0x85eb_ca6b,
            precomputed_chunk_count,
        };

        if let Some(dbias) = dbias {
            return self
                .module
                .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_with_bias(
                    args, dbias,
                );
        }

        self.module
            .fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(args)?;
        Ok(false)
    }
}
