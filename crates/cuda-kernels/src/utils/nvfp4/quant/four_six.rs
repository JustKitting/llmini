use cuda_core::DriverError;

use super::args::{
    Nvfp4QuantArgs, Nvfp4QuantPaddedArgs, Nvfp4QuantRowwiseArgs, Nvfp4QuantTransposePaddedArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::{four_six_grid_config, four_six_rowwise_pow2, four_six_transpose_tiled_config};

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
