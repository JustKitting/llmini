#[path = "ms_eden/quantize.rs"]
mod quantize;

use cuda_core::DriverError;

use crate::launch::grid_x_config;

use self::quantize::QuantizeContext;
use super::{
    LINEAR_BIAS_THREADS_PER_BLOCK, LinearBackwardDeviceScaleArgs, LinearBackwardModule,
    LinearBackwardMsEdenArgs, bias,
};

impl LinearBackwardModule {
    pub fn backward_ms_eden(
        &self,
        mut args: LinearBackwardMsEdenArgs<'_, '_, '_>,
    ) -> Result<(), DriverError> {
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

        self.backward_device_scale_tma(
            LinearBackwardDeviceScaleArgs {
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
            },
            scratch.tma,
        )
    }
}
