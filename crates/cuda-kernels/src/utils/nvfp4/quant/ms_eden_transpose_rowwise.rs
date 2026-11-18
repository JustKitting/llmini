use cuda_core::DriverError;

use super::args::RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_rowwise_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden_rowwise_transpose
            .padded
            .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scales,
                args.out_fp4,
                args.out_scales,
                args.out_chunk_amax,
                &*args.out_global_scale,
                pack.chunk_count,
                args.source_rows,
                args.source_cols,
                args.dst_row_len,
                QUARTET_MS_EDEN_SCALE_OVERRIDE,
                args.sign_seed,
                args.scale_seed,
            )
    }

    pub fn rowwise_nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        super::launcher::validate_csf_scales(args.out_scales, args.source_cols, args.dst_row_len)?;
        if args.input.bytes.len() < (args.source_rows as usize) * (args.source_cols as usize) / 2
            || args.input.scales.len()
                < crate::nvfp4::csf::rowwise_encoded_len(
                    args.input.global_scales.len(),
                    args.source_cols as usize,
                )
            || args.input.global_scales.len() < args.source_rows as usize
        {
            return Err(DriverError(
                cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }

        self.derive_rowwise_nvfp4_transpose_global_scale(&mut args)?;
        self.ms_eden_csf.ms_eden_rowwise_nvfp4_transpose_csf_kernel(
            args.stream,
            crate::launch::launch_config(
                (args.dst_row_len / 128, args.source_cols.div_ceil(64), 1),
                256,
            ),
            args.input.bytes,
            args.input.scales,
            args.input.global_scales,
            args.out_fp4,
            args.out_scales,
            &*args.out_global_scale,
            args.source_rows,
            args.source_cols,
            (args.input.bytes.len() * 2 / args.source_cols as usize) as u32,
            args.sign_seed,
            args.scale_seed,
            QUARTET_MS_EDEN_SCALE_OVERRIDE,
        )
    }
}
