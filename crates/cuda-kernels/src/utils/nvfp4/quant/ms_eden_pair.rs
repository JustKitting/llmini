use cuda_core::DriverError;

use cuda_core::DeviceBuffer;

use super::args::MsEdenPairDeviceScaleQuantArgs;
use super::launcher::Nvfp4QuantModule;

impl Nvfp4QuantModule {
    pub fn fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
        &self,
        args: MsEdenPairDeviceScaleQuantArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        self.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_impl(
            args, None,
        )
        .map(|_| ())
    }

    pub fn fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_with_bias(
        &self,
        args: MsEdenPairDeviceScaleQuantArgs<'_, '_>,
        out_bias: &mut DeviceBuffer<f32>,
    ) -> Result<bool, DriverError> {
        self.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_impl(
            args,
            Some(out_bias),
        )
    }

    fn fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax_impl(
        &self,
        args: MsEdenPairDeviceScaleQuantArgs<'_, '_>,
        out_bias: Option<&mut DeviceBuffer<f32>>,
    ) -> Result<bool, DriverError> {
        super::launcher::validate_csf_scales(args.out_scales, args.row_count, args.dst_row_len)?;
        super::launcher::validate_csf_scales(
            args.transpose_out_scales,
            args.src_row_len,
            args.transpose_dst_row_len,
        )?;
        let chunk_count = if let Some(chunk_count) = args.precomputed_chunk_count {
            chunk_count
        } else {
            self.tensor_chunk_amax_f32(
                args.stream,
                args.x,
                &mut *args.out_chunk_amax,
                args.row_count * args.src_row_len,
            )?
        };

        self.quartet_backward_ms_eden_global_scale_from_chunks(
            args.stream,
            &*args.out_chunk_amax,
            &mut *args.out_global_scale,
            chunk_count,
        )?;

        let mut empty_bias;
        let fused_bias = out_bias.is_some();
        let bias = if let Some(b) = out_bias {
            unsafe {
                cuda_core::memory::memset_d8_async(
                    b.cu_deviceptr(),
                    0,
                    b.len() * 4,
                    args.stream.cu_stream(),
                )?;
            }
            b
        } else {
            empty_bias = DeviceBuffer::zeroed(args.stream, 0)?;
            &mut empty_bias
        };
        self.ms_eden_csf.ms_eden_fp32_pair_csf_kernel(
            args.stream,
            crate::launch::launch_config(
                ((args.row_count / 128) * (args.dst_row_len / 128), 1, 1),
                512,
            ),
            args.x,
            args.out_fp4,
            args.out_scales,
            args.transpose_out_fp4,
            args.transpose_out_scales,
            bias,
            &*args.out_global_scale,
            args.row_count,
            args.src_row_len,
            args.dst_row_len,
            args.transpose_dst_row_len,
            args.sign_seed,
            args.scale_seed,
            args.transpose_scale_seed,
            args.scale_override,
        )?;
        Ok(fused_bias)
    }
}
