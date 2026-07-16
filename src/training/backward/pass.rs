use gpt2_nvfp4::{Gpt2BackwardArgs, Gpt2BackwardSeeds, gpt2_backward};
use rust_kernels_cuda::residual::ResidualGradAccumulateArgs;

use crate::training::next_latent::{
    NextLatBackwardArgs, NextLatBackwardSeeds, backward as next_latent_backward,
};
use crate::{
    AppResult,
    training::{TokenBatch, Trainer},
};

impl Trainer {
    pub(super) fn enqueue_backward(&mut self, batch: &TokenBatch) -> AppResult {
        let stream = self.runtime.stream.as_ref();
        {
            let saved = self.buffers.tape.saved(
                &batch.tokens,
                batch.batch_size as u32,
                batch.seq_len as u32,
                batch.token_count as u32,
                &self.buffers.logits,
            );
            let weights = self.uploaded.backward_weights();
            let backward = self.buffers.backward.parts();
            let scratch = self
                .buffers
                .scratch
                .scratch(&mut self.buffers.normalized, &mut self.buffers.mlp_act);

            next_latent_backward(NextLatBackwardArgs {
                stream,
                next_latent: &self.runtime.next_latent,
                linear: &self.runtime.linear,
                quant: &self.runtime.quant,
                layer_norm: &self.runtime.layer_norm_backward,
                weights: &self.uploaded.next_latent,
                forward: &self.buffers.next_latent,
                grads: &mut self.buffers.next_latent_grads,
                scratch: &mut self.buffers.next_latent_scratch,
                row_count: batch.token_count as u32,
                seeds: NextLatBackwardSeeds::from_rng(&mut self.rng),
            })?;

            gpt2_backward(Gpt2BackwardArgs {
                stream,
                modules: self.runtime.backward_modules(),
                saved,
                weights,
                targets: &batch.targets,
                losses: backward.losses,
                extra_final_normalized_grad: Some(&self.buffers.next_latent_grads.d_current_states),
                d_lm_head_weight: backward.d_lm_head_weight,
                grads: backward.grads,
                scratch,
                seeds: Gpt2BackwardSeeds::from_rng(&mut self.rng),
            })?;
        }
        self.runtime
            .residual
            .grad_accumulate(ResidualGradAccumulateArgs {
                stream,
                branch: self.buffers.next_latent.losses(),
                out: &mut self.buffers.backward.losses,
                len: batch.token_count as u32,
            })?;
        Ok(())
    }
}
