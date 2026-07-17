use cuda_core::DriverError;

use super::args::Nvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::{MsEdenPackGrid, ms_eden_rowwise_transpose_tiled_config};
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub fn nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale(
        &self,
        mut args: Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden_nvfp4_transpose
            .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scale,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
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

    pub fn nvfp4_transpose_to_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        mut args: Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.derive_nvfp4_transpose_global_scale(&mut args)?;
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        if pack.is_exact() {
            if args.source_rows == args.dst_row_len
                && args.source_rows != 0
                && args.source_rows.is_multiple_of(32)
                && args.source_cols.is_multiple_of(8)
                && args.source_cols.is_power_of_two()
            {
                let config =
                    ms_eden_rowwise_transpose_tiled_config(args.source_rows, args.source_cols);
                let source_cols_shift = args.source_cols.trailing_zeros();
                let chunks_per_row = args.source_rows / 32;
                if chunks_per_row.is_power_of_two() {
                    return self
                        .ms_eden_nvfp4_transpose
                        .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_kernel(
                            args.stream,
                            config,
                            args.input.bytes,
                            args.input.scales,
                            args.input.global_scale,
                            args.out_fp4,
                            args.out_scales,
                            args.out_global_scales,
                            &*args.out_global_scale,
                            source_cols_shift,
                            chunks_per_row.trailing_zeros(),
                            QUARTET_MS_EDEN_SCALE_OVERRIDE,
                            args.sign_seed,
                            args.scale_seed,
                        );
                }
                return self
                    .ms_eden_nvfp4_transpose
                    .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_row_mul_kernel(
                        args.stream,
                        config,
                        args.input.bytes,
                        args.input.scales,
                        args.input.global_scale,
                        args.out_fp4,
                        args.out_scales,
                        args.out_global_scales,
                        &*args.out_global_scale,
                        source_cols_shift,
                        chunks_per_row,
                        QUARTET_MS_EDEN_SCALE_OVERRIDE,
                        args.sign_seed,
                        args.scale_seed,
                    );
            }
            return self
                .ms_eden_nvfp4_transpose
                .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.input.bytes,
                    args.input.scales,
                    args.input.global_scale,
                    args.out_fp4,
                    args.out_scales,
                    args.out_global_scales,
                    &*args.out_global_scale,
                    args.source_rows,
                    args.source_cols,
                    args.dst_row_len,
                    QUARTET_MS_EDEN_SCALE_OVERRIDE,
                    args.sign_seed,
                    args.scale_seed,
                );
        }

        self.ms_eden_nvfp4_transpose
            .nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
                args.stream,
                pack.config(),
                args.input.bytes,
                args.input.scales,
                args.input.global_scale,
                args.out_fp4,
                args.out_scales,
                args.out_global_scales,
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
}
