use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_VOCAB_DIM};
use rust_kernels_cuda::loss::CrossEntropyArgs;

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
        self.forward_step(batch)?;

        self.collect_token_losses(batch)
    }

    pub(in crate::training) fn eval_token_losses_for_passes(
        &mut self,
        batch: &TokenBatch,
        passes: usize,
    ) -> AppResult<Vec<f32>> {
        self.forward_for_passes(batch, passes)?;
        self.collect_token_losses(batch)
    }

    fn collect_token_losses(&mut self, batch: &TokenBatch) -> AppResult<Vec<f32>> {
        let stream = self.runtime.stream.as_ref();
        self.runtime.loss.cross_entropy(CrossEntropyArgs {
            stream,
            logits: &self.buffers.logits,
            targets: &batch.targets,
            losses: &mut self.buffers.backward.losses,
            dlogits: &mut self.buffers.backward.dlogits,
            dlogits_row_amax: &mut self.buffers.normalized_amax,
            token_count: batch.token_count as u32,
            vocab_size: GPT2_VOCAB_DIM,
        })?;

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
