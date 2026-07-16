use cuda_core::DriverError;
use gpt2_nvfp4::{GPT2_EMBEDDING_DIM, NEXTLAT_HIDDEN_DIM, NEXTLAT_INPUT_DIM};

use super::backward::NextLatBackwardArgs;
use super::backward_linear_call::{LinearCall, run_linear};

pub(super) fn output_projection_backward(
    args: &mut NextLatBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    run_linear(LinearCall {
        linear: args.linear,
        quant: args.quant,
        stream: args.stream,
        e: &args.forward.d_predicted,
        input: args.forward.act2_quant.rowwise(),
        weight: &args.weights.output_projection,
        scratch: &mut args.scratch.output_projection,
        dinput: &mut args.grads.d_act2,
        dweight: &mut args.grads.d_output_projection_weight,
        dbias: &mut args.grads.d_output_projection_bias,
        row_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN_DIM,
        output_dim: GPT2_EMBEDDING_DIM,
        sign_seed: args.seeds.output_sign,
        scale_seed: args.seeds.output_scale,
    })
}

pub(super) fn transition_backward(
    args: &mut NextLatBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    run_linear(LinearCall {
        linear: args.linear,
        quant: args.quant,
        stream: args.stream,
        e: &args.grads.d_pre2,
        input: args.forward.act1_quant.rowwise(),
        weight: &args.weights.transition,
        scratch: &mut args.scratch.transition,
        dinput: &mut args.grads.d_act2,
        dweight: &mut args.grads.d_transition_weight,
        dbias: &mut args.grads.d_transition_bias,
        row_count: args.row_count,
        input_dim: NEXTLAT_HIDDEN_DIM,
        output_dim: NEXTLAT_HIDDEN_DIM,
        sign_seed: args.seeds.transition_sign,
        scale_seed: args.seeds.transition_scale,
    })
}

pub(super) fn input_projection_backward(
    args: &mut NextLatBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    run_linear(LinearCall {
        linear: args.linear,
        quant: args.quant,
        stream: args.stream,
        e: &args.grads.d_pre2,
        input: args.forward.input_quant.rowwise(),
        weight: &args.weights.input_projection,
        scratch: &mut args.scratch.input_projection,
        dinput: &mut args.grads.d_act2,
        dweight: &mut args.grads.d_input_projection_weight,
        dbias: &mut args.grads.d_input_projection_bias,
        row_count: args.row_count,
        input_dim: NEXTLAT_INPUT_DIM,
        output_dim: NEXTLAT_HIDDEN_DIM,
        sign_seed: args.seeds.input_sign,
        scale_seed: args.seeds.input_scale,
    })
}
