use gpt2_nvfp4::{GPT2_VOCAB_SIZE, Gpt2ForwardArgs, TokenEmbeddingArgs};

use super::next_latent::{
    NextLatForwardArgs, forward as next_latent_forward, loss_weight as next_latent_loss_weight,
};
use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn forward_step(&mut self, batch: &TokenBatch) -> AppResult<TrainStats> {
        self.forward_for_passes(batch, self.evaluation_passes())
    }

    pub(super) fn evaluation_passes(&self) -> usize {
        if self.latent_feedback.is_some() {
            super::latent_feedback::EVAL_PASSES
        } else {
            1
        }
    }

    pub(in crate::training) fn forward_for_passes(
        &mut self,
        batch: &TokenBatch,
        passes: usize,
    ) -> AppResult<TrainStats> {
        self.run_forward(batch, passes, true)
    }

    pub(super) fn forward_logits_for_passes(
        &mut self,
        batch: &TokenBatch,
        passes: usize,
    ) -> AppResult<TrainStats> {
        self.run_forward(batch, passes, false)
    }

    fn run_forward(
        &mut self,
        batch: &TokenBatch,
        passes: usize,
        include_next_latent: bool,
    ) -> AppResult<TrainStats> {
        self.ensure_ready()?;
        if passes == 0 || (self.latent_feedback.is_none() && passes != 1) {
            return Err("invalid feedback pass count".into());
        }
        if let Some(f) = &mut self.latent_feedback {
            f.begin(
                &self.runtime,
                &mut self.buffers.muon_tables,
                &self.buffers.grad_clip,
                passes,
                batch.token_count as u32,
            )?;
        }
        let mut stats = TrainStats::default();
        for pass in 0..passes {
            stats = self.forward_pass(batch, pass > 0, pass + 1 == passes, include_next_latent)?;
        }
        stats.feedback_passes = passes;
        Ok(stats)
    }

    fn forward_pass(
        &mut self,
        batch: &TokenBatch,
        apply_feedback: bool,
        readout: bool,
        include_next_latent: bool,
    ) -> AppResult<TrainStats> {
        let stream = self.runtime.stream.as_ref();
        let buffers = &mut self.buffers;
        let uploaded = &self.uploaded;

        let engram = &mut self.engram;
        let runtime = &self.runtime;
        let feedback = &mut self.latent_feedback;
        let feedback_weights = &uploaded.latent_feedback;
        let mhc = std::cell::RefCell::new(&mut self.mhc);
        let mhc_weights = &uploaded.mhc;
        let has_mhc = mhc.borrow().is_some();
        let mlp_projection_observer: Option<&dyn gpt2_nvfp4::MlpProjectionObserver> = None;
        #[cfg(test)]
        let mlp_projection_observer = self
            .bounded_projection_audit
            .as_ref()
            .map(|a| a as &dyn gpt2_nvfp4::MlpProjectionObserver);
        let args = Gpt2ForwardArgs {
            value_rms: &self.runtime.value_rms,
            value_projection_observer: {
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
            mlp_paired: &mut buffers.scratch.mlp_paired,
            mlp_projection_observer,
            embeddings: TokenEmbeddingArgs {
                module: &self.runtime.embedding,
                stream,
                tokens: &batch.tokens,
                token_embedding: uploaded.token_embedding.device(),
                batch_size: batch.batch_size as u32,
                seq_len: batch.seq_len as u32,
                row_count: batch.token_count as u32,
                residual: &mut buffers.residual,
                normalized: &mut buffers.normalized,
                normalized_amax: &mut buffers.normalized_amax,
                mean: &mut buffers.mean,
                inv_std: &mut buffers.inv_std,
            },
            attention_module: &self.runtime.attention,
            canon_module: &self.runtime.canon,
            attention_tc_module: &self.runtime.f16_tc_matmul,
            quant_module: &self.runtime.quant,
            layer_norm_module: &self.runtime.layer_norm,
            mlp_module: &self.runtime.mlp,
            lm_head_module: &self.runtime.lm_head,
            tma_module: &self.runtime.tma_gemm,
            tma_pad: &self.runtime.tma_pad,
            tma_descriptors: &mut buffers.forward_tma.descriptors,
            tma_weight_bytes_padded: &mut buffers.forward_tma.weight_bytes_padded,
            hidden_nvfp4: buffers.hidden_nvfp4.scratch(),
            attention_tc_scratch: buffers.scratch.attention_core.forward_tc(),
            mlp_activation_nvfp4: buffers.mlp_activation_nvfp4.scratch(),
            attention: std::array::from_fn(|i| {
                uploaded.blocks[i].attention_tensors(
                    uploaded.xsa_alphas.device(),
                    (i * gpt2_nvfp4::GPT2_N_HEAD) as u32,
                )
            }),
            block_ln_1: std::array::from_fn(|i| uploaded.blocks[i].ln_1.tensors()),
            block_canon_a: std::array::from_fn(|i| uploaded.blocks[i].canon_a.tensors()),
            block_ln_2: std::array::from_fn(|i| uploaded.blocks[i].ln_2.tensors()),
            block_canon_c: std::array::from_fn(|i| uploaded.blocks[i].canon_c.tensors()),
            mlp: std::array::from_fn(|i| uploaded.blocks[i].mlp_tensors()),
            ln_f: uploaded.ln_f.tensors(),
            attention_qkv: &mut buffers.qkv,
            attention_value_residual: &mut buffers.value_residual,
            attention_log_sum_exp: &mut buffers.log_sum_exp,
            mlp_activation: &mut buffers.mlp_act,
            mlp_route_scores: &mut buffers.mlp_route_scores,
            mlp_route_masks: &mut buffers.mlp_route_masks,
            logits: &mut buffers.logits,
            tape: readout.then(|| buffers.tape.tape()),
        };
        let mut hook = |index: usize,
                        hidden: &mut gpt2_nvfp4::HiddenStateDevice<'_>|
         -> Result<(), cuda_core::DriverError> {
            if index == 0 && apply_feedback {
                if let (Some(f), Some(w)) = (feedback.as_mut(), feedback_weights.as_ref()) {
                    f.mix(runtime, w, hidden)?;
                }
            }
            if let (Some(e), Some(w)) = (engram.as_mut(), uploaded.engram.as_ref()) {
                if w.streams() == Some(4) {
                    let mut state = mhc.borrow_mut();
                    let m = state.as_mut().expect("validated four-stream backbone");
                    e.forward_at_streams(
                        index,
                        runtime,
                        w,
                        &batch.tokens,
                        &mut m.current,
                        hidden.row_count,
                        hidden.seq_len,
                    )?;
                } else {
                    e.forward_at(index, runtime, w, &batch.tokens, hidden)?;
                }
            }
            if index == gpt2_nvfp4::GPT2_N_LAYER && !readout {
                if let Some(f) = feedback.as_mut() {
                    f.prepare_input(runtime, hidden)?;
                }
            }
            Ok(())
        };
        let mut boundary = |index, before, hidden: &mut gpt2_nvfp4::HiddenStateDevice<'_>| {
            let mut state = mhc.borrow_mut();
            if let (Some(m), Some(w)) = (state.as_mut(), mhc_weights.as_ref()) {
                m.forward_boundary(index, before, runtime, w, hidden)
            } else {
                Ok(false)
            }
        };
        let hidden = if has_mhc {
            self.model
                .forward_with_boundaries(args, &mut hook, readout, &mut boundary)?
        } else if readout {
            self.model.forward_with_hook(args, &mut hook)?
        } else {
            self.model.forward_hidden_with_hook(args, &mut hook)?
        };

        if readout && include_next_latent {
            next_latent_forward(NextLatForwardArgs {
                stream,
                embedding: &self.runtime.embedding,
                layer_norm: &self.runtime.layer_norm,
                quant: &self.runtime.quant,
                tma: &self.runtime.tma_gemm,
                next_latent: &self.runtime.next_latent,
                token_embedding: uploaded.token_embedding.device(),
                weights: &uploaded.next_latent,
                targets: &batch.targets,
                current_states: hidden.normalized,
                buffers: &mut buffers.next_latent,
                batch_size: batch.batch_size as u32,
                seq_len: batch.seq_len as u32,
                row_count: batch.token_count as u32,
                lambda: next_latent_loss_weight(),
            })?;
        }

        Ok(TrainStats {
            tokens: batch.token_count,
            logits: batch.token_count * GPT2_VOCAB_SIZE,
            finite: true,
            ..TrainStats::default()
        })
    }
}
