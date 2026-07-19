use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::training::OptimizerTrace;
use crate::training::grads::BackwardBuffers;
use crate::training::next_latent::NextLatGradBuffers;
use crate::training::optimizer::OptimizerScratch;
use crate::training::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;

use super::adam::AdamUpdate;
use super::ember::EmberUpdate;
use super::layer_norm::update_layer_norm_timed;
use super::next_latent::{NextLatUpdateArgs, update_next_latent};
use super::timed_ms;

pub(super) struct BaseAdamUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub optimizer: &'a OptimizerModule,
    pub uploaded: &'a mut UploadedModel,
    pub grads: &'a BackwardBuffers,
    pub next_latent_grads: &'a NextLatGradBuffers,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut OptimizerStateBuffers,
    pub step: u32,
    pub average_coefficient: f32,
    pub grad_scale: f32,
    pub trace: &'a mut OptimizerTrace,
}

pub(super) fn update_base_adam(args: BaseAdamUpdateArgs<'_>) -> Result<(), DriverError> {
    let token_embedding_ms = match &mut args.state.token_embedding {
        super::super::optimizer_state::TokenEmbeddingState::Adam(state) => {
            let mut adam = AdamUpdate::new(
                args.stream,
                args.optimizer,
                args.scratch,
                args.step,
                args.average_coefficient,
                args.grad_scale,
            );
            adam.update_timed(
                &mut args.uploaded.token_embedding,
                &args.grads.d_lm_head_weight,
                state,
            )?
        }
        super::super::optimizer_state::TokenEmbeddingState::Ember(state) => EmberUpdate::new(
            args.stream,
            args.optimizer,
            args.step,
            args.average_coefficient,
            args.grad_scale,
        )
        .update_timed(
            &args.uploaded.token_embedding,
            &args.grads.d_lm_head_weight,
            state,
        )?,
    };

    let (xsa_alphas_ms, final_norm_ms) = {
        let mut adam = AdamUpdate::new(
            args.stream,
            args.optimizer,
            args.scratch,
            args.step,
            args.average_coefficient,
            args.grad_scale,
        );
        let xsa_alphas_ms = if gpt2_nvfp4::exclusive_self_attention_enabled() {
            adam.update_timed_with_weight_decay(
                &mut args.uploaded.xsa_alphas,
                &args.grads.d_xsa_alphas,
                &mut args.state.xsa_alphas,
                0.0,
            )?
        } else {
            0.0
        };
        let final_norm_ms = update_layer_norm_timed(
            &mut adam,
            &mut args.uploaded.ln_f,
            &args.grads.final_norm,
            &mut args.state.ln_f,
        )?;
        (xsa_alphas_ms, final_norm_ms)
    };

    let next_latent_ms = timed_ms(|| {
        update_next_latent(NextLatUpdateArgs {
            stream: args.stream,
            optimizer: args.optimizer,
            weights: &mut args.uploaded.next_latent,
            grads: args.next_latent_grads,
            scratch: args.scratch,
            state: &mut args.state.next_latent,
            step: args.step,
            average_coefficient: args.average_coefficient,
            grad_scale: args.grad_scale,
        })
    })?;

    args.trace.token_embedding_ms = token_embedding_ms;
    args.trace.final_norm_ms = final_norm_ms;
    args.trace.adam_ms += token_embedding_ms + xsa_alphas_ms + final_norm_ms + next_latent_ms;
    Ok(())
}
