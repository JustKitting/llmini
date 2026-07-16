use cuda_core::DriverError;
use gpt2_nvfp4::NEXTLAT_INPUT_DIM;
use rust_kernels_cuda::layer_norm_backward::{
    LayerNormBackwardInputF32Args, LayerNormBackwardParamF32Args,
};

use super::backward::NextLatBackwardArgs;

pub(super) fn layer_norm_backward(
    args: &mut NextLatBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    args.layer_norm
        .backward_params_f32(LayerNormBackwardParamF32Args {
            stream: args.stream,
            residual: &args.forward.concat,
            d_normalized: &args.grads.d_act2,
            mean: &args.forward.mean,
            inv_std: &args.forward.inv_std,
            d_weight: &mut args.grads.d_norm_weight,
            d_bias: &mut args.grads.d_norm_bias,
            row_count: args.row_count,
            embedding_dim: NEXTLAT_INPUT_DIM,
        })?;
    args.layer_norm
        .backward_input_f32(LayerNormBackwardInputF32Args {
            stream: args.stream,
            residual: &args.forward.concat,
            d_normalized: &args.grads.d_act2,
            mean: &args.forward.mean,
            inv_std: &args.forward.inv_std,
            weight: args.weights.norm.weight.device(),
            d_residual: &mut args.grads.d_pre2,
            row_count: args.row_count,
            embedding_dim: NEXTLAT_INPUT_DIM,
        })
}
