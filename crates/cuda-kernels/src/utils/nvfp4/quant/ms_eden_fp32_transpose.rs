use cuda_core::DriverError;

use super::args::MsEdenTransposeDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;
use super::shape::MsEdenPackGrid;

impl Nvfp4QuantModule {
    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale(
        &self,
        args: MsEdenTransposeDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let pack = MsEdenPackGrid::for_elements(args.source_cols * args.dst_row_len);
        self.ms_eden_fp32_transpose
            .fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel(
                args.stream,
                pack.config(),
                args.x,
                args.out_fp4,
                args.out_scales,
                args.out_chunk_amax,
                args.global_scale,
                pack.chunk_count,
                args.source_rows,
                args.source_cols,
                args.dst_row_len,
                args.scale_override,
                args.sign_seed,
                args.scale_seed,
            )
    }

    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax(
        &self,
        args: MsEdenTransposeDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        super::launcher::validate_csf_scales(args.out_scales, args.source_cols, args.dst_row_len)?;
        self.ms_eden_csf.ms_eden_fp32_transpose_csf_kernel(
            args.stream,
            crate::launch::launch_config(
                (args.dst_row_len / 128, args.source_cols.div_ceil(32), 1),
                256,
            ),
            args.x,
            args.out_fp4,
            args.out_scales,
            args.global_scale,
            args.source_rows,
            args.source_cols,
            args.dst_row_len,
            args.sign_seed,
            args.scale_seed,
            args.scale_override,
        )
    }
}
