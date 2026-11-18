use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::args::{
    Nvfp4TransposeMsEdenDeviceScaleQuantArgs, QuartetBackwardMsEdenDeviceScaleQuantArgs,
    RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use super::launcher::Nvfp4QuantModule;
use super::shape::grid_config;
use crate::launch::launch_config;

impl Nvfp4QuantModule {
    pub fn quartet_backward_ms_eden_global_scale_from_chunks(
        &self,
        stream: &CudaStream,
        chunk_amax: &DeviceBuffer<f32>,
        out_global_scale: &mut DeviceBuffer<f32>,
        chunk_count: u32,
    ) -> Result<(), DriverError> {
        self.ms_eden_amax
            .quartet_backward_ms_eden_global_scale_from_chunks_kernel(
                stream,
                grid_config(1),
                chunk_amax,
                out_global_scale,
                chunk_count,
            )
    }

    pub(super) fn derive_rowwise_nvfp4_transpose_global_scale(
        &self,
        args: &mut RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        debug_assert!(args.source_cols.is_multiple_of(16));
        self.ms_eden_amax
            .quartet_backward_ms_eden_global_scale_from_rowwise_four_six_kernel(
                args.stream,
                launch_config((1, 1, 1), 1024),
                args.input.scales,
                args.input.global_scales,
                &mut *args.out_global_scale,
                args.source_rows,
                args.source_cols / 16,
            )
    }

    pub(super) fn derive_nvfp4_transpose_global_scale(
        &self,
        args: &mut Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let element_count = args.source_rows * args.source_cols;
        debug_assert!(element_count.is_multiple_of(16));
        self.ms_eden_amax
            .quartet_backward_ms_eden_global_scale_from_four_six_kernel(
                args.stream,
                grid_config(1),
                args.input.scales,
                args.input.global_scale,
                &mut *args.out_global_scale,
                element_count / 16,
            )
    }

    pub(super) fn derive_fp32_quartet_backward_ms_eden_global_scale(
        &self,
        args: &mut QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        let chunk_count = self.tensor_chunk_amax_f32(
            args.stream,
            args.x,
            &mut *args.out_chunk_amax,
            args.row_count * args.src_row_len,
        )?;

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )
    }
}
