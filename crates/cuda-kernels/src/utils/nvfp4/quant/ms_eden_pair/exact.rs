use cuda_core::{DeviceBuffer, DriverError};

use super::super::args::MsEdenPairDeviceScaleQuantArgs;
use super::super::launcher::Nvfp4QuantModule;
use super::super::shape::{Fp32PairNoPad, MsEdenPackGrid, grid_config};

impl Nvfp4QuantModule {
    pub(super) fn launch_pair_exact_bias_no_chunk_amax(
        &self,
        args: &mut MsEdenPairDeviceScaleQuantArgs<'_, '_>,
        out_bias: &mut DeviceBuffer<f32>,
        row_pack: MsEdenPackGrid,
        transpose_pack: MsEdenPackGrid,
    ) -> Option<Result<(), DriverError>> {
        if !row_pack.is_exact() || !transpose_pack.is_exact() {
            return None;
        }

        let no_pad = Fp32PairNoPad::new(
            args.row_count,
            args.src_row_len,
            args.dst_row_len,
            args.transpose_dst_row_len,
        )?;
        if !args.row_count.is_multiple_of(32) || !args.src_row_len.is_multiple_of(8) {
            return None;
        }

        let grid = grid_config(row_pack.grid_dim + args.src_row_len / 8);
        if let Some((chunks_per_row_shift, transpose_chunks_per_row_shift)) = no_pad.pow2_shifts() {
            return Some(
                self.ms_eden_fp32_pair
                    .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_tiled_bias_kernel(
                        args.stream,
                        grid,
                        args.x,
                        &mut *args.out_fp4,
                        &mut *args.out_scales,
                        &mut *args.out_global_scales,
                        &mut *args.transpose_out_fp4,
                        &mut *args.transpose_out_scales,
                        &mut *args.transpose_out_global_scales,
                        out_bias,
                        &*args.out_global_scale,
                        row_pack.grid_dim,
                        args.src_row_len,
                        chunks_per_row_shift,
                        transpose_chunks_per_row_shift,
                        args.scale_override,
                        args.sign_seed,
                        args.scale_seed,
                        args.transpose_scale_seed,
                    ),
            );
        }

        Some(
            self.ms_eden_fp32_pair
                .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_tiled_bias_kernel(
                    args.stream,
                    grid,
                    args.x,
                    &mut *args.out_fp4,
                    &mut *args.out_scales,
                    &mut *args.out_global_scales,
                    &mut *args.transpose_out_fp4,
                    &mut *args.transpose_out_scales,
                    &mut *args.transpose_out_global_scales,
                    out_bias,
                    &*args.out_global_scale,
                    row_pack.grid_dim,
                    args.src_row_len,
                    no_pad.chunks_per_row,
                    no_pad.transpose_chunks_per_row,
                    args.scale_override,
                    args.sign_seed,
                    args.scale_seed,
                    args.transpose_scale_seed,
                ),
        )
    }

    pub(super) fn launch_pair_exact_no_chunk_amax(
        &self,
        args: &mut MsEdenPairDeviceScaleQuantArgs<'_, '_>,
        row_pack: MsEdenPackGrid,
        transpose_pack: MsEdenPackGrid,
    ) -> Option<Result<(), DriverError>> {
        if !row_pack.is_exact() || !transpose_pack.is_exact() {
            return None;
        }

        let grid = grid_config(row_pack.grid_dim + transpose_pack.grid_dim);
        if let Some(no_pad) = Fp32PairNoPad::new(
            args.row_count,
            args.src_row_len,
            args.dst_row_len,
            args.transpose_dst_row_len,
        ) {
            if let Some((chunks_per_row_shift, transpose_chunks_per_row_shift)) =
                no_pad.pow2_shifts()
            {
                return Some(
                    self.ms_eden_fp32_pair
                        .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_tiled_kernel(
                            args.stream,
                            grid,
                            args.x,
                            &mut *args.out_fp4,
                            &mut *args.out_scales,
                            &mut *args.out_global_scales,
                            &mut *args.transpose_out_fp4,
                            &mut *args.transpose_out_scales,
                            &mut *args.transpose_out_global_scales,
                            &*args.out_global_scale,
                            row_pack.grid_dim,
                            args.src_row_len,
                            chunks_per_row_shift,
                            transpose_chunks_per_row_shift,
                            args.scale_override,
                            args.sign_seed,
                            args.scale_seed,
                            args.transpose_scale_seed,
                        ),
                );
            }

            return Some(
                self.ms_eden_fp32_pair
                    .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_tiled_kernel(
                        args.stream,
                        grid,
                        args.x,
                        &mut *args.out_fp4,
                        &mut *args.out_scales,
                        &mut *args.out_global_scales,
                        &mut *args.transpose_out_fp4,
                        &mut *args.transpose_out_scales,
                        &mut *args.transpose_out_global_scales,
                        &*args.out_global_scale,
                        row_pack.grid_dim,
                        args.src_row_len,
                        no_pad.chunks_per_row,
                        no_pad.transpose_chunks_per_row,
                        args.scale_override,
                        args.sign_seed,
                        args.scale_seed,
                        args.transpose_scale_seed,
                    ),
            );
        }

        Some(
            self.ms_eden_fp32_pair
                .fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
                    args.stream,
                    grid,
                    args.x,
                    &mut *args.out_fp4,
                    &mut *args.out_scales,
                    &mut *args.out_global_scales,
                    &mut *args.transpose_out_fp4,
                    &mut *args.transpose_out_scales,
                    &mut *args.transpose_out_global_scales,
                    &*args.out_global_scale,
                    row_pack.grid_dim,
                    args.row_count,
                    args.src_row_len,
                    args.dst_row_len,
                    args.transpose_dst_row_len,
                    args.scale_override,
                    args.sign_seed,
                    args.scale_seed,
                    args.transpose_scale_seed,
                ),
        )
    }
}
