#[path = "ms_eden/quantize.rs"]
mod quantize;

use cuda_core::{DeviceBuffer, DriverError, sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE};

use crate::launch::grid_x_config;
use crate::nvfp4::Nvfp4RowwiseDeviceTensor;

use self::quantize::QuantizeContext;
use super::{
    LINEAR_BIAS_THREADS_PER_BLOCK, LinearBackwardDeviceScaleArgs, LinearBackwardModule,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenWeightArgs, LinearBackwardRoute, bias,
};

impl LinearBackwardModule {
    pub fn backward_ms_eden(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        self.backward_ms_eden_impl(args, None, None).map(|_| ())
    }

    pub fn backward_ms_eden_weight(
        &self,
        args: LinearBackwardMsEdenWeightArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
        let quantize = QuantizeContext::new(
            args.stream,
            args.quant_module,
            args.token_count,
            args.input_dim,
            args.output_dim,
            args.sign_seed,
            args.scale_seed,
        );
        let mut scratch = args.scratch;
        // Use the same paired pack and transpose seed for dW as the full
        // backward path, even though the other orientation is unused.
        quantize.error_pair(args.e, &mut scratch, None, None)?;
        quantize.input_transpose(args.input_t, &mut scratch.input_t_h)?;
        let error_t = Nvfp4RowwiseDeviceTensor::new(
            &*scratch.e_t_h.bytes,
            &*scratch.e_t_h.scales,
            &*scratch.e_h.global_scale,
        );
        self.backward_weight_device_scale_tma(
            args.stream,
            error_t,
            scratch.input_t_h.device_scale_mma_weight(),
            args.dweight,
            args.token_count,
            args.input_dim,
            args.output_dim,
            scratch.tma.descriptors,
            None,
        )
    }

    pub fn backward_ms_eden_routed(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        route: LinearBackwardRoute<'_>,
    ) -> Result<(), DriverError> {
        self.backward_ms_eden_impl(args, None, Some(route))
            .map(|_| ())
    }

    pub fn backward_ms_eden_relu2_backward_f16(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        pre_activation: &DeviceBuffer<u16>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
    ) -> Result<u32, DriverError> {
        self.backward_ms_eden_impl(args, Some((pre_activation, output_chunk_amax)), None)?
            .ok_or(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE))
    }

    pub fn backward_ms_eden_relu2_backward_f16_routed(
        &self,
        args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        pre_activation: &DeviceBuffer<u16>,
        output_chunk_amax: &mut DeviceBuffer<f32>,
        route: LinearBackwardRoute<'_>,
    ) -> Result<u32, DriverError> {
        self.backward_ms_eden_impl(args, Some((pre_activation, output_chunk_amax)), Some(route))?
            .ok_or(DriverError(cudaError_enum_CUDA_ERROR_INVALID_VALUE))
    }

    fn backward_ms_eden_impl(
        &self,
        mut args: LinearBackwardMsEdenArgs<'_, '_, '_>,
        relu2_backward_f16: Option<(&DeviceBuffer<u16>, &mut DeviceBuffer<f32>)>,
        route: Option<LinearBackwardRoute<'_>>,
    ) -> Result<Option<u32>, DriverError> {
        let quantize = QuantizeContext::new(
            args.stream,
            args.quant_module,
            args.token_count,
            args.input_dim,
            args.output_dim,
            args.sign_seed,
            args.scale_seed,
        );
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
            // The paired transpose uses the same derived global scalar.
            e_t_h: crate::nvfp4::Nvfp4RowwiseDeviceTensor::new(
                &*scratch.e_t_h.bytes,
                &*scratch.e_t_h.scales,
                &*scratch.e_h.global_scale,
            ),
            input_t_h: scratch.input_t_h.device_scale_mma_weight(),
            dinput: args.dinput,
            dweight: args.dweight,
            token_count: args.token_count,
            input_dim: args.input_dim,
            output_dim: args.output_dim,
        };
        match (relu2_backward_f16, route) {
            (Some((pre_activation, output_chunk_amax)), Some(route)) => self
                .backward_device_scale_tma_relu2_backward_f16_routed(
                    device_args,
                    scratch.tma,
                    pre_activation,
                    output_chunk_amax,
                    route,
                )
                .map(Some),
            (Some((pre_activation, output_chunk_amax)), None) => self
                .backward_device_scale_tma_relu2_backward_f16(
                    device_args,
                    scratch.tma,
                    pre_activation,
                    output_chunk_amax,
                )
                .map(Some),
            (None, Some(route)) => {
                self.backward_device_scale_tma_routed(device_args, scratch.tma, route)?;
                Ok(None)
            }
            (None, None) => {
                self.backward_device_scale_tma(device_args, scratch.tma)?;
                Ok(None)
            }
        }
    }
}
