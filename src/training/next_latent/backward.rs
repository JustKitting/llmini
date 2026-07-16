use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_EMBEDDING_DIM, NEXTLAT_HIDDEN_DIM};
use rust_kernels_cuda::next_latent::{
    NextLatConcatBackwardArgs, NextLatGeluBackwardArgs, NextLatModule,
};

use super::backward_linear::{
    input_projection_backward, output_projection_backward, transition_backward,
};
use super::backward_norm::layer_norm_backward;

mod types;

pub use types::{NextLatBackwardArgs, NextLatBackwardSeeds};

pub fn backward(mut args: NextLatBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    output_projection_backward(&mut args)?;
    gelu_backward(
        args.next_latent,
        args.stream,
        &args.forward.act1,
        &args.grads.d_act2,
        &mut args.grads.d_pre2,
        args.row_count * NEXTLAT_HIDDEN_DIM,
    )?;
    transition_backward(&mut args)?;
    gelu_backward(
        args.next_latent,
        args.stream,
        &args.forward.normalized,
        &args.grads.d_act2,
        &mut args.grads.d_pre2,
        args.row_count * NEXTLAT_HIDDEN_DIM,
    )?;
    input_projection_backward(&mut args)?;
    layer_norm_backward(&mut args)?;
    args.next_latent.concat_backward(NextLatConcatBackwardArgs {
        stream: args.stream,
        d_concat: &args.grads.d_pre2,
        d_predicted: &args.forward.d_predicted,
        d_next_token_embeddings: &mut args.grads.d_next_token_embeddings,
        d_current_states: &mut args.grads.d_current_states,
        row_count: args.row_count,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })
}

fn gelu_backward(
    module: &NextLatModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    d_out: &DeviceBuffer<f32>,
    d_input: &mut DeviceBuffer<f32>,
    len: u32,
) -> Result<(), DriverError> {
    module.gelu_backward(NextLatGeluBackwardArgs {
        stream,
        input,
        d_out,
        d_input,
        len,
    })
}
