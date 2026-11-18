use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::linear_backward::MsEdenOperandScratch;
use crate::nvfp4_quant::{
    MsEdenDeviceScaleQuantArgs, Nvfp4QuantModule, QuartetBackwardMsEdenDeviceScaleQuantArgs,
};
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

pub(super) struct QuantizeOperandArgs<'a, 'operand, 'scratch> {
    pub(super) stream: &'a CudaStream,
    pub(super) x: &'a DeviceBuffer<f32>,
    pub(super) operand: &'operand mut MsEdenOperandScratch<'scratch>,
    pub(super) row_count: u32,
    pub(super) src_row_len: u32,
    pub(super) dst_row_len: u32,
    pub(super) sign_seed: u32,
    pub(super) scale_seed: u32,
    pub(super) precomputed_chunk_count: Option<u32>,
}

pub(super) fn quantize_operand(
    module: &Nvfp4QuantModule,
    args: QuantizeOperandArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let operand = args.operand;
    if let Some(chunk_count) = args.precomputed_chunk_count {
        module.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*operand.chunk_amax,
            &mut *operand.global_scale,
            chunk_count,
        )?;

        return module.fp32_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
            MsEdenDeviceScaleQuantArgs {
                stream: args.stream,
                x: args.x,
                out_fp4: operand.bytes,
                out_scales: operand.scales,

                out_chunk_amax: operand.chunk_amax,
                global_scale: &*operand.global_scale,
                row_count: args.row_count,
                src_row_len: args.src_row_len,
                dst_row_len: args.dst_row_len,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: args.sign_seed,
                scale_seed: args.scale_seed,
            },
        );
    }

    module.fp32_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream: args.stream,
            x: args.x,
            out_fp4: operand.bytes,
            out_scales: operand.scales,

            out_chunk_amax: operand.chunk_amax,
            out_global_scale: operand.global_scale,
            row_count: args.row_count,
            src_row_len: args.src_row_len,
            dst_row_len: args.dst_row_len,
            sign_seed: args.sign_seed,
            scale_seed: args.scale_seed,
        },
    )
}
