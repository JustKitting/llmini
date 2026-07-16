use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{
    Nvfp4QuantArgs, Nvfp4QuantPaddedArgs, Nvfp4QuantPairTransposeExactArgs, Nvfp4QuantRowwiseArgs,
    Nvfp4QuantRowwiseDerivedAmaxArgs, Nvfp4QuantTransposePaddedArgs, RowAmaxArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::{four_six_grid_config, four_six_rowwise_pow2, four_six_transpose_tiled_config};
use crate::launch::grid_x_config;
use crate::nvfp4_tma_matmul::cute::Sm120ScaleLayout;

const SCALE_OVERRIDE: f32 = 1.0;

impl Nvfp4QuantModule {
    pub fn fp32_to_nvfp4_four_six(&self, args: Nvfp4QuantArgs<'_, '_>) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six(Nvfp4QuantRowwiseArgs {
            stream: args.stream,
            x: args.x,
            amax: args.amax,
            out_fp4: args.out_fp4,
            out_scales: args.out_scales,
            out_global_scale: args.out_global_scale,
            group_count: args.group_count,
            row_len: 0,
        })
    }

    pub fn fp32_to_nvfp4_four_six_rowwise(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        if four_six_rowwise_pow2(args.row_len, args.group_count) {
            return self.launch_fp32_to_nvfp4_four_six_rowwise_pow2(args);
        }

        self.launch_fp32_to_nvfp4_four_six(args)
    }

    pub fn fp32_to_nvfp4_four_six_rowwise_derived_amax(
        &self,
        args: Nvfp4QuantRowwiseDerivedAmaxArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let Nvfp4QuantRowwiseDerivedAmaxArgs {
            stream,
            x,
            amax,
            out_fp4,
            out_scales,
            out_global_scale,
            row_count,
            row_len,
        } = args;
        let group_count = row_count * row_len / 16;
        if four_six_rowwise_pow2(row_len, group_count) {
            return self
                .four_six
                .fp32_to_nvfp4_four_six_rowwise_derived_amax_pow2_kernel(
                    stream,
                    grid_x_config(row_count, 256),
                    x,
                    amax,
                    out_fp4,
                    out_scales,
                    out_global_scale,
                    row_count,
                    row_len,
                    SCALE_OVERRIDE,
                );
        }

        self.row_amax_f32(RowAmaxArgs {
            stream,
            x,
            out: &mut *amax,
            row_count,
            row_len,
        })?;
        self.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x,
            amax: &*amax,
            out_fp4,
            out_scales,
            out_global_scale,
            group_count,
            row_len,
        })
    }

    pub fn fp32_to_nvfp4_four_six_padded(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let padded_elements = args.padded_rows * args.padded_cols;
        assert!(padded_elements.is_multiple_of(16));
        if args.rows == args.padded_rows && args.cols == args.padded_cols {
            return self.four_six.fp32_to_nvfp4_four_six_exact_kernel(
                args.stream,
                four_six_grid_config(padded_elements / 16),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                SCALE_OVERRIDE,
            );
        }

        self.four_six.fp32_to_nvfp4_four_six_padded_kernel(
            args.stream,
            four_six_grid_config(padded_elements / 16),
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.rows,
            args.cols,
            args.padded_cols,
            SCALE_OVERRIDE,
        )
    }

    pub fn fp32_pair_to_nvfp4_four_six_exact_pow2_tiled(
        &self,
        args: Nvfp4QuantPairTransposeExactArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.source_rows.is_power_of_two());
        assert!(args.source_rows.is_multiple_of(16));
        assert!(args.source_cols.is_multiple_of(64));
        self.four_six
            .fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_kernel(
                args.stream,
                four_six_transpose_tiled_config(args.source_rows, args.source_cols),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.transpose_out_fp4,
                args.transpose_out_scales,
                args.transpose_out_global_scale,
                args.source_rows,
                args.source_cols,
                SCALE_OVERRIDE,
            )
    }

    pub fn fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_packed_scales(
        &self,
        args: Nvfp4QuantPairTransposeExactArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert!(args.source_rows.is_power_of_two());
        assert!(args.source_rows.is_multiple_of(Sm120ScaleLayout::MN_BLOCK));
        assert!(args.source_cols.is_multiple_of(Sm120ScaleLayout::MN_BLOCK));
        assert!(
            args.out_scales.len()
                >= Sm120ScaleLayout::packed_len(args.source_rows, args.source_cols)
        );
        assert!(
            args.transpose_out_scales.len()
                >= Sm120ScaleLayout::packed_len(args.source_cols, args.source_rows)
        );
        self.four_six
            .fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_packed_scales_kernel(
                args.stream,
                four_six_transpose_tiled_config(args.source_rows, args.source_cols),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.transpose_out_fp4,
                args.transpose_out_scales,
                args.transpose_out_global_scale,
                args.source_rows,
                args.source_cols,
                SCALE_OVERRIDE,
            )
    }

    pub fn rebase_four_six_global_scale_sqrt_bound(
        &self,
        stream: &CudaStream,
        original_amax: &DeviceBuffer<f32>,
        sqrt_bound_amax: &DeviceBuffer<f32>,
        out_global_scale: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.four_six
            .four_six_rebase_sqrt_bound_global_scale_kernel(
                stream,
                grid_x_config(1, 32),
                original_amax,
                sqrt_bound_amax,
                out_global_scale,
            )
    }

    pub fn fp32_to_nvfp4_four_six_exact_bounded_amax(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six_exact_bounded_amax(args, false)
    }

    pub fn fp32_to_nvfp4_four_six_exact_lazy_bounded_amax(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six_exact_bounded_amax(args, true)
    }

    pub fn fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales(args, false)
    }

    pub fn fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales(args, true)
    }

    fn launch_fp32_to_nvfp4_four_six_exact_bounded_amax(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
        apply_bound: bool,
    ) -> Result<(), DriverError> {
        assert_eq!(args.rows, args.padded_rows);
        assert_eq!(args.cols, args.padded_cols);
        let elements = args.rows * args.cols;
        assert!(elements.is_multiple_of(16));
        self.four_six
            .fp32_to_nvfp4_four_six_exact_bounded_amax_kernel(
                args.stream,
                four_six_grid_config(elements / 16),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                apply_bound as u32,
            )
    }

    fn launch_fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales(
        &self,
        args: Nvfp4QuantPaddedArgs<'_, '_>,
        apply_bound: bool,
    ) -> Result<(), DriverError> {
        assert_eq!(args.rows, args.padded_rows);
        assert_eq!(args.cols, args.padded_cols);
        assert!(args.rows.is_multiple_of(Sm120ScaleLayout::MN_BLOCK));
        assert!(args.cols.is_multiple_of(Sm120ScaleLayout::K_ATOM));
        assert!(args.out_scales.len() >= Sm120ScaleLayout::packed_len(args.rows, args.cols));
        let elements = args.rows * args.cols;
        self.four_six
            .fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales_kernel(
                args.stream,
                four_six_grid_config(elements / 16),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.rows,
                args.cols,
                apply_bound as u32,
            )
    }

    pub fn fp32_transpose_to_nvfp4_four_six_padded(
        &self,
        args: Nvfp4QuantTransposePaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let padded_elements = args.padded_rows * args.padded_cols;
        assert!(padded_elements.is_multiple_of(16));
        if args.source_cols == args.padded_rows && args.source_rows == args.padded_cols {
            if args.source_rows.is_power_of_two()
                && args.source_rows.is_multiple_of(16)
                && args.source_cols.is_multiple_of(64)
            {
                return self
                    .four_six
                    .fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_kernel(
                        args.stream,
                        four_six_transpose_tiled_config(args.source_rows, args.source_cols),
                        args.x,
                        args.amax,
                        args.out_fp4,
                        args.out_scales,
                        args.out_global_scale,
                        args.source_rows,
                        args.source_cols,
                        SCALE_OVERRIDE,
                    );
            }

            if args.source_rows.is_power_of_two() {
                return self
                    .four_six
                    .fp32_transpose_to_nvfp4_four_six_exact_pow2_kernel(
                        args.stream,
                        four_six_grid_config(padded_elements / 16),
                        args.x,
                        args.amax,
                        args.out_fp4,
                        args.out_scales,
                        args.out_global_scale,
                        args.source_rows.trailing_zeros(),
                        args.source_rows - 1,
                        args.source_cols,
                        SCALE_OVERRIDE,
                    );
            }

            return self.four_six.fp32_transpose_to_nvfp4_four_six_exact_kernel(
                args.stream,
                four_six_grid_config(padded_elements / 16),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.source_rows,
                args.source_cols,
                SCALE_OVERRIDE,
            );
        }

        self.four_six
            .fp32_transpose_to_nvfp4_four_six_padded_kernel(
                args.stream,
                four_six_grid_config(padded_elements / 16),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.source_rows,
                args.source_cols,
                args.padded_cols,
                SCALE_OVERRIDE,
            )
    }

    pub fn fp32_transpose_to_nvfp4_four_six_exact_packed_scales(
        &self,
        args: Nvfp4QuantTransposePaddedArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.source_cols, args.padded_rows);
        assert_eq!(args.source_rows, args.padded_cols);
        assert!(args.source_rows.is_power_of_two());
        assert!(args.source_rows.is_multiple_of(Sm120ScaleLayout::K_ATOM));
        assert!(args.source_cols.is_multiple_of(Sm120ScaleLayout::MN_BLOCK));
        assert!(
            args.out_scales.len()
                >= Sm120ScaleLayout::packed_len(args.source_cols, args.source_rows)
        );
        self.four_six
            .fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_packed_scales_kernel(
                args.stream,
                four_six_transpose_tiled_config(args.source_rows, args.source_cols),
                args.x,
                args.amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.source_rows,
                args.source_cols,
                SCALE_OVERRIDE,
            )
    }

    pub fn fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
        &self,
        args: Nvfp4QuantTransposePaddedArgs<'_, '_>,
        sqrt_bound_amax: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
            args,
            sqrt_bound_amax,
            false,
        )
    }

    pub fn fp32_transpose_to_nvfp4_four_six_exact_lazy_sqrt_bounded_amax(
        &self,
        args: Nvfp4QuantTransposePaddedArgs<'_, '_>,
        sqrt_bound_amax: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.launch_fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
            args,
            sqrt_bound_amax,
            true,
        )
    }

    fn launch_fp32_transpose_to_nvfp4_four_six_exact_sqrt_bounded_amax(
        &self,
        args: Nvfp4QuantTransposePaddedArgs<'_, '_>,
        sqrt_bound_amax: &DeviceBuffer<f32>,
        apply_bound: bool,
    ) -> Result<(), DriverError> {
        assert_eq!(args.source_cols, args.padded_rows);
        assert_eq!(args.source_rows, args.padded_cols);
        assert!(args.source_rows.is_power_of_two());
        assert!(args.source_rows.is_multiple_of(16));
        assert!(args.source_cols.is_multiple_of(64));
        self.four_six
            .fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_sqrt_bounded_amax_kernel(
                args.stream,
                four_six_transpose_tiled_config(args.source_rows, args.source_cols),
                args.x,
                args.amax,
                sqrt_bound_amax,
                args.out_fp4,
                args.out_scales,
                args.out_global_scale,
                args.source_rows,
                args.source_cols,
                apply_bound as u32,
            )
    }

    fn launch_fp32_to_nvfp4_four_six(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.four_six.fp32_to_nvfp4_four_six_kernel(
            args.stream,
            four_six_grid_config(args.group_count),
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.row_len,
            SCALE_OVERRIDE,
        )
    }

    fn launch_fp32_to_nvfp4_four_six_rowwise_pow2(
        &self,
        args: Nvfp4QuantRowwiseArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.four_six.fp32_to_nvfp4_four_six_rowwise_pow2_kernel(
            args.stream,
            four_six_grid_config(args.group_count),
            args.x,
            args.amax,
            args.out_fp4,
            args.out_scales,
            args.out_global_scale,
            args.row_len.trailing_zeros(),
            args.row_len - 1,
            SCALE_OVERRIDE,
        )
    }
}
