use cuda_core::DriverError;

use crate::linear_backward::{
    LinearBackwardInputTranspose, LinearBackwardWeightTranspose, MsEdenOperandScratch,
};
use crate::nvfp4_quant::{
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};

use super::context::QuantizeContext;
use super::operand::{QuantizeOperandArgs, quantize_operand};

impl<'a> QuantizeContext<'a> {
    pub(in crate::linear_backward::ms_eden) fn weight_transpose(
        &self,
        weight_t: LinearBackwardWeightTranspose<'_>,
        operand: &mut MsEdenOperandScratch<'_>,
    ) -> Result<(), DriverError> {
        match weight_t {
            LinearBackwardWeightTranspose::Fp32(weight_t) => quantize_operand(
                self.module,
                QuantizeOperandArgs {
                    stream: self.stream,
                    x: weight_t,
                    operand,
                    row_count: self.input_dim,
                    src_row_len: self.output_dim,
                    dst_row_len: self.output_k,
                    sign_seed: self.sign_seed,
                    scale_seed: self.scale_seed ^ 0x9e37_79b9,
                    precomputed_chunk_count: None,
                },
            ),
            LinearBackwardWeightTranspose::Nvfp4(weight) => self
                .module
                .nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                    Nvfp4TransposeMsEdenDeviceScaleQuantArgs {
                        stream: self.stream,
                        input: weight,
                        out_fp4: &mut *operand.bytes,
                        out_scales: &mut *operand.scales,

                        out_chunk_amax: &mut *operand.chunk_amax,
                        out_global_scale: &mut *operand.global_scale,
                        source_rows: self.output_dim,
                        source_cols: self.input_dim,
                        dst_row_len: self.output_k,
                        sign_seed: self.sign_seed,
                        scale_seed: self.scale_seed ^ 0x9e37_79b9,
                    },
                ),
        }
    }

    pub(in crate::linear_backward::ms_eden) fn input_transpose(
        &self,
        input_t: LinearBackwardInputTranspose<'_>,
        operand: &mut MsEdenOperandScratch<'_>,
    ) -> Result<(), DriverError> {
        match input_t {
            LinearBackwardInputTranspose::Fp32(input_t) => quantize_operand(
                self.module,
                QuantizeOperandArgs {
                    stream: self.stream,
                    x: input_t,
                    operand,
                    row_count: self.input_dim,
                    src_row_len: self.token_count,
                    dst_row_len: self.token_k,
                    sign_seed: self.sign_seed,
                    scale_seed: self.scale_seed ^ 0xc2b2_ae35,
                    precomputed_chunk_count: None,
                },
            ),
            LinearBackwardInputTranspose::RowwiseNvfp4(input_t) => self
                .module
                .rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
                    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
                        stream: self.stream,
                        input: input_t,
                        out_fp4: &mut *operand.bytes,
                        out_scales: &mut *operand.scales,

                        out_chunk_amax: &mut *operand.chunk_amax,
                        out_global_scale: &mut *operand.global_scale,
                        source_rows: self.token_count,
                        source_cols: self.input_dim,
                        dst_row_len: self.token_k,
                        sign_seed: self.sign_seed,
                        scale_seed: self.scale_seed ^ 0xc2b2_ae35,
                    },
                ),
        }
    }
}
