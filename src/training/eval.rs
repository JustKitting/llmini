use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_VOCAB_DIM};
use rust_kernels_cuda::loss::{CrossEntropyArgs, CrossEntropyParams};

use super::{TokenBatch, Trainer};
use crate::AppResult;

impl Trainer {
    pub fn eval_loss(&mut self, batch: &TokenBatch) -> AppResult<f32> {
        let losses = self.eval_token_losses(batch)?;
        Ok(losses.iter().sum::<f32>() / losses.len() as f32)
    }

    pub(in crate::training) fn eval_token_losses(
        &mut self,
        batch: &TokenBatch,
    ) -> AppResult<Vec<f32>> {
        self.eval_token_losses_for_passes(batch, self.evaluation_passes())
    }

    pub(in crate::training) fn eval_token_losses_for_passes(
        &mut self,
        batch: &TokenBatch,
        passes: usize,
    ) -> AppResult<Vec<f32>> {
        self.forward_logits_for_passes(batch, passes)?;
        self.collect_token_losses(batch)
    }

    fn collect_token_losses(&mut self, batch: &TokenBatch) -> AppResult<Vec<f32>> {
        let stream = self.runtime.stream.as_ref();
        let params = CrossEntropyParams {
            token_count: batch.token_count as u32,
            vocab_size: GPT2_VOCAB_DIM,
        };
        if !self.runtime.loss.try_cross_entropy_loss(
            stream,
            &self.buffers.logits,
            &batch.targets,
            &mut self.buffers.backward.losses,
            params,
        )? {
            self.runtime.loss.cross_entropy(CrossEntropyArgs {
                stream,
                logits: &self.buffers.logits,
                targets: &batch.targets,
                losses: &mut self.buffers.backward.losses,
                dlogits: &mut self.buffers.backward.dlogits,
                dlogits_row_amax: &mut self.buffers.normalized_amax,
                token_count: params.token_count,
                vocab_size: params.vocab_size,
            })?;
        }

        let losses = self.buffers.backward.losses.to_host_vec(stream)?;
        Ok(losses[..batch.token_count].to_vec())
    }

    pub fn eval_loss_windows(&mut self, windows: &[u16], window_count: usize) -> AppResult<f32> {
        let window_len = GPT2_SEQ_LEN + 1;
        let needed = window_count * window_len;
        if windows.len() < needed {
            return Err(format!(
                "validation window has {} tokens, needs {}",
                windows.len(),
                needed
            )
            .into());
        }

        let mut loss_sum = 0.0;
        let mut token_count = 0usize;
        let mut window = 0;
        while window < window_count {
            let batch_windows = (window_count - window).min(GPT2_BATCH_SIZE);
            let start = window * window_len;
            let end = start + batch_windows * window_len;
            let batch = self.batch_from_windows(&windows[start..end], batch_windows)?;
            let loss = self.eval_loss(&batch)?;
            loss_sum += loss * batch.token_count as f32;
            token_count += batch.token_count;
            window += batch_windows;
        }

        Ok(loss_sum / token_count as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::TokenDataLoader;
    use sha2::{Digest, Sha256};

    #[ignore = "requires the saved native Rational checkpoint and GPU0"]
    #[test]
    fn read_only_scoring_matches_the_full_forward_and_loss() -> AppResult {
        let checkpoint = std::path::PathBuf::from(
            std::env::var_os("NVFP4_READONLY_SCORE_CHECKPOINT")
                .ok_or("set NVFP4_READONLY_SCORE_CHECKPOINT")?,
        );
        assert_eq!(GPT2_BATCH_SIZE, 8);
        let mut trainer = Trainer::new(0x47505432)?;
        trainer.load_model(&checkpoint)?;
        let validation = TokenDataLoader::fineweb_validation_tokens()?;
        let stream = trainer.runtime.stream.clone();
        let mut scored_losses = Sha256::new();
        for batch_size in [4, 8] {
            let tokens = validation
                .iter()
                .copied()
                .cycle()
                .take(batch_size * (GPT2_SEQ_LEN + 1))
                .collect::<Vec<_>>();
            let batch = trainer.batch_from_windows(&tokens, batch_size)?;
            trainer.forward_step(&batch)?;
            trainer.runtime.loss.cross_entropy(CrossEntropyArgs {
                stream: &stream,
                logits: &trainer.buffers.logits,
                targets: &batch.targets,
                losses: &mut trainer.buffers.backward.losses,
                dlogits: &mut trainer.buffers.backward.dlogits,
                dlogits_row_amax: &mut trainer.buffers.normalized_amax,
                token_count: batch.token_count as u32,
                vocab_size: GPT2_VOCAB_DIM,
            })?;
            let reference = trainer.buffers.backward.losses.to_host_vec(&stream)?;
            let next_latent = trainer.buffers.next_latent.losses();
            unsafe {
                cuda_core::memory::memset_d8_async(
                    next_latent.cu_deviceptr(),
                    255,
                    next_latent.num_bytes(),
                    stream.cu_stream(),
                )?;
            }
            let actual = trainer.eval_token_losses(&batch)?;
            assert_eq!(actual.len(), batch.token_count);
            for (&loss, &full) in actual.iter().zip(&reference) {
                assert!(loss.is_finite());
                assert_eq!(loss.to_bits(), full.to_bits());
                scored_losses.update(loss.to_bits().to_le_bytes());
            }
            assert!(
                trainer
                    .buffers
                    .next_latent
                    .losses()
                    .to_host_vec(&stream)?
                    .iter()
                    .all(|loss| loss.to_bits() == u32::MAX)
            );
        }
        assert_eq!(trainer.buffers.optimizer_state.step, 0);
        println!(
            "read_only_scoring_token_loss_sha256={:x}",
            scored_losses.finalize()
        );
        Ok(())
    }
}
