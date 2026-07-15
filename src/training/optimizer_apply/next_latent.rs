use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::training::next_latent::NextLatGradBuffers;
use crate::training::optimizer::OptimizerScratch;
use crate::training::optimizer_state::NextLatState;
use crate::upload::UploadedNextLat;

use super::adam::{AdamUpdate, next_latent_adam_learning_rate};

pub(super) struct NextLatUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub optimizer: &'a OptimizerModule,
    pub weights: &'a mut UploadedNextLat,
    pub grads: &'a NextLatGradBuffers,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut NextLatState,
    pub step: u32,
    pub average_coefficient: f32,
    pub grad_scale: f32,
}

pub(super) fn update_next_latent(args: NextLatUpdateArgs<'_>) -> Result<(), DriverError> {
    let NextLatUpdateArgs {
        stream,
        optimizer,
        weights,
        grads,
        scratch,
        state,
        step,
        average_coefficient,
        grad_scale,
    } = args;
    let learning_rate = next_latent_adam_learning_rate(step);
    let mut adam = AdamUpdate::with_learning_rate(
        stream,
        optimizer,
        scratch,
        step,
        average_coefficient,
        grad_scale,
        learning_rate,
    );

    adam.update(
        &mut weights.norm.weight,
        &grads.d_norm_weight,
        &mut state.norm.weight,
    )?;
    adam.update(
        &mut weights.norm.bias,
        &grads.d_norm_bias,
        &mut state.norm.bias,
    )?;
    adam.update(
        &mut weights.input_projection.bias,
        &grads.d_input_projection_bias,
        &mut state.input_projection.bias,
    )?;
    adam.update(
        &mut weights.transition.bias,
        &grads.d_transition_bias,
        &mut state.transition.bias,
    )?;
    adam.update(
        &mut weights.output_projection.bias,
        &grads.d_output_projection_bias,
        &mut state.output_projection.bias,
    )
}
