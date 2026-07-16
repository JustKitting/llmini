#[path = "ms_eden/quantize.rs"]
mod quantize;

use cuda_core::{DeviceBuffer, DriverError, sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE};

use crate::launch::grid_x_config;

use self::quantize::QuantizeContext;
use super::{
    LINEAR_BIAS_THREADS_PER_BLOCK, LinearBackwardDeviceScaleArgs, LinearBackwardModule,
    LinearBackwardMsEdenArgs, bias,
};

impl LinearBackwardModule {
    pub fn backward_ms_eden(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        self.backward_ms_eden_impl(args, None).map(|_| ())
    }

    pub fn backward_ms_eden_relu2_backward_f16(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        pre_activation: &DeviceBuffer<u16>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, DriverError> {
        self.backward_ms_eden_impl(args, Some((pre_activation, output_chunk_amax)))?
            .ok_or(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE))
    }

    fn backward_ms_eden_impl(
        &self,
        mut args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        relu2_backward_f16: Option<(&DeviceBuffer<u16>, &mut DeviceBuffer<f32>)>,
    ) -> Result<Option<u32>, DriverError> {
        let quantize = QuantizeContext::for_args(&args);
        let mut scratch = args.scratch;
        let bias_fused = quantize.error_pair(
            args.e,
            &mut scratch,
            args.precomputed_e_amax_chunks,
            args.dbias.as_deref_mut(),
        )?;

        if !bias_fused && let Some(dbias) = args.dbias {
            self.module.bias.linear_bias_grad_kernel(
                args.stream,
                grid_x_config(
                    bias::grid_dim(args.output_dim),
                    LINEAR_BIAS_THREADS_PER_BLOCK,
                ),
                args.e,
                dbias,
                args.token_count,
                args.output_dim,
            )?;
        }

        quantize.weight_transpose(args.weight_t, &mut scratch.weight_t_h)?;
        quantize.input_transpose(args.input_t, &mut scratch.input_t_h)?;

        let device_args = LinearBackwardDeviceScaleArgs {
            stream: args.stream,
            e_h: scratch.e_h.rowwise(),
            weight_t_h: scratch.weight_t_h.device_scale_mma_weight(),
            e_t_h: scratch.e_t_h.rowwise(),
            input_t_h: scratch.input_t_h.device_scale_mma_weight(),
            dinput: args.dinput,
            dweight: args.dweight,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
        };
        if let Some((pre_activation, output_chunk_amax)) = relu2_backward_f16 {
            self.backward_device_scale_tma_relu2_backward_f16(
                device_args,
                scratch.tma,
                pre_activation,
                output_chunk_amax,
            )
            .map(Some)
        } else {
            self.backward_device_scale_tma(device_args, scratch.tma)?;
            Ok(None)
        }
    }
}
