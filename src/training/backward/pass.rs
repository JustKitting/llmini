use gpt2_nvfp4::gpt2_backward_at_depth;
use gpt2_nvfp4::{Gpt2BackwardArgs, Gpt2BackwardSeeds};
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
        let active_depth = self.active_depth();
        self.buffers.optimizer_state.invalidate_scheduled_values();
        super::super::symexp_lin::prepare_weight_gradients(
            &self.runtime,
            &self.buffers.muon_tables,
            &self.buffers.grad_clip,
            super::super::learning_rate::schedule_free_beta(
                self.buffers.optimizer_state.next_step(),
            ),
        )?;
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
            let mlp_projection_observer: Option<&dyn gpt2_nvfp4::MlpProjectionObserver> = None;
            #[cfg(test)]
            let mlp_projection_observer = self
                .bounded_projection_audit
                .as_ref()
                .map(|a| a as &dyn gpt2_nvfp4::MlpProjectionObserver);
            let scratch = self.buffers.scratch.scratch(
                &mut self.buffers.normalized,
                &mut self.buffers.value_residual,
                &mut self.buffers.mlp_act,
                mlp_projection_observer,
                {
                    #[cfg(test)]
                    {
                        self.value_projection_audit
                            .as_ref()
                            .map(|a| a as &dyn gpt2_nvfp4::ValueProjectionObserver)
                    }
                    #[cfg(not(test))]
                    {
                        None
                    }
                },
            );

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

            let engram = &mut self.engram;
            let engram_weights = &self.uploaded.engram;
            let runtime = &self.runtime;
            let mhc = std::cell::RefCell::new(&mut self.mhc);
            let mhc_weights = &self.uploaded.mhc;
            gpt2_backward_at_depth(
                active_depth,
                Gpt2BackwardArgs {
                    stream,
                    modules: self.runtime.backward_modules(),
                    saved,
                    weights,
                    targets: &batch.targets,
                    losses: backward.losses,
                    extra_final_normalized_grad: Some(
                        &self.buffers.next_latent_grads.d_current_states,
                    ),
                    d_lm_head_weight: backward.d_lm_head_weight,
                    grads: backward.grads,
                    scratch,
                    seeds: Gpt2BackwardSeeds::from_rng(&mut self.rng),
                },
                &mut |index, gradient| {
                    if let (Some(e), Some(w)) = (engram.as_mut(), engram_weights.as_ref()) {
                        if w.streams() == Some(4) {
                            let mut state = mhc.borrow_mut();
                            let m = state.as_mut().expect("validated four-stream backbone");
                            e.backward_at(index, runtime, w, &mut m.current)?;
                            // The authoritative four-stream gradient was
                            // updated in place. The base ND scratch is stale.
                            return Ok(false);
                        }
                        return e.backward_at(index, runtime, w, gradient);
                    }
                    Ok(false)
                },
                &mut |index, before, gradient| {
                    let mut state = mhc.borrow_mut();
                    if let (Some(m), Some(w)) = (state.as_mut(), mhc_weights.as_ref()) {
                        let unit = if index == active_depth * 2 {
                            crate::training::mhc::BOUNDARIES
                        } else {
                            index
                        };
                        m.backward_boundary(unit, before, runtime, w, gradient)
                    } else {
                        Ok(false)
                    }
                },
            )?;
        }
        if let Some(f) = &mut self.latent_feedback {
            f.backward(
                &self.runtime,
                &self.buffers.backward.d_embedding_residual,
                batch.seq_len,
            )?;
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
