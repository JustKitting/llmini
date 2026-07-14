use cuda_core::DriverError;

use super::super::args::RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs;
use super::super::launcher::Nvfp4QuantModule;
use super::super::shape::{
    MsEdenPackGrid, RowwiseTransposeNoPad, ms_eden_rowwise_transpose_tiled_config,
};
use crate::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;

impl Nvfp4QuantModule {
    pub(super) fn launch_rowwise_transpose_exact_no_chunk_amax(
        &self,
        args: &mut RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
        pack: MsEdenPackGrid,
    ) -> Option<Result<(), DriverError>> {
        if !pack.is_exact() {
            return None;
        }

        if let Some(no_pad) =
            RowwiseTransposeNoPad::new(args.source_rows, args.source_cols, args.dst_row_len)
        {
            if let Some(source_cols_shift) = no_pad.source_cols_shift() {
                if args.source_rows.is_multiple_of(32) && args.source_cols.is_multiple_of(8) {
                    return Some(
                        self.ms_eden_rowwise_transpose
                            .no_pad
                            .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_kernel(
                                args.stream,
                                ms_eden_rowwise_transpose_tiled_config(
                                    args.source_rows,
                                    args.source_cols,
                                ),
                                args.input.bytes,
                                args.input.scales,
                                args.input.global_scales,
                                &mut *args.out_fp4,
                                &mut *args.out_scales,
                                &mut *args.out_global_scales,
                                &*args.out_global_scale,
                                source_cols_shift,
                                no_pad.chunks_per_row_shift,
                                QUARTET_MS_EDEN_SCALE_OVERRIDE,
                                args.sign_seed,
                                args.scale_seed,
                            ),
                    );
                }

                return Some(
                    self.ms_eden_rowwise_transpose
                        .no_pad
                        .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_kernel(
                            args.stream,
                            pack.config(),
                            args.input.bytes,
                            args.input.scales,
                            args.input.global_scales,
                            &mut *args.out_fp4,
                            &mut *args.out_scales,
                            &mut *args.out_global_scales,
                            &*args.out_global_scale,
                            source_cols_shift,
                            no_pad.chunks_per_row_shift,
                            QUARTET_MS_EDEN_SCALE_OVERRIDE,
                            args.sign_seed,
                            args.scale_seed,
                        ),
                );
            }

            return Some(
                self.ms_eden_rowwise_transpose
                    .no_pad
                    .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
                        args.stream,
                        pack.config(),
                        args.input.bytes,
                        args.input.scales,
                        args.input.global_scales,
                        &mut *args.out_fp4,
                        &mut *args.out_scales,
                        &mut *args.out_global_scales,
                        &*args.out_global_scale,
                        no_pad.source_cols,
                        no_pad.chunks_per_row_shift,
                        QUARTET_MS_EDEN_SCALE_OVERRIDE,
                        args.sign_seed,
                        args.scale_seed,
                    ),
            );
        }

        Some(
            self.ms_eden_rowwise_transpose
                .padded
                .rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    pack.config(),
                    args.input.bytes,
                    args.input.scales,
                    args.input.global_scales,
                    &mut *args.out_fp4,
                    &mut *args.out_scales,
                    &mut *args.out_global_scales,
                    &*args.out_global_scale,
                    args.source_rows,
                    args.source_cols,
                    args.dst_row_len,
                    QUARTET_MS_EDEN_SCALE_OVERRIDE,
                    args.sign_seed,
                    args.scale_seed,
                ),
        )
    }
}
