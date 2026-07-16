use std::time::Instant;

mod pass;

use super::optimizer_apply::{WeightUpdateArgs, apply_weight_updates};
use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub(in crate::training) fn materialize_training_weights(&mut self) -> AppResult<()> {
        super::schedule_free::materialize_training_weights(
            self.runtime.stream.as_ref(),
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
        )?;
        Ok(())
    }

    pub(in crate::training) fn materialize_evaluation_weights(&mut self) -> AppResult<()> {
        super::schedule_free::materialize_evaluation_weights(
            self.runtime.stream.as_ref(),
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
        )?;
        Ok(())
    }

    pub fn train_step(&mut self, batch: &TokenBatch, sync_loss: bool) -> AppResult<TrainStats> {
        self.materialize_training_weights()?;

        let forward_start = Instant::now();
        let mut stats = self.forward_step(batch)?;
        stats.forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

        let backward_start = Instant::now();
        self.enqueue_backward(batch)?;
        stats.backward_enqueue_ms = backward_start.elapsed().as_secs_f64() * 1000.0;

        if sync_loss {
            self.sync_loss(batch, &mut stats)?;
        }
        let observed_loss = sync_loss.then_some(stats.loss);

        let optimizer_start = Instant::now();
        let stream = self.runtime.stream.as_ref();
        let updates = apply_weight_updates(WeightUpdateArgs {
            stream,
            runtime: &self.runtime,
            batch,
            uploaded: &mut self.uploaded,
            grads: &mut self.buffers.backward,
            next_latent_grads: &self.buffers.next_latent_grads,
            observed_loss,
            scratch: &mut self.buffers.optimizer,
            state: &mut self.buffers.optimizer_state,
            muon: &mut self.buffers.muon,
            muon_tables: &self.buffers.muon_tables,
            tape: &self.buffers.tape,
            qk_norm_max: self.buffers.scratch.attention_core.qk_norm_max(),
            grad_clip: &mut self.buffers.grad_clip,
        })?;
        stats.optimizer = updates.trace;
        stats.optimizer_ms = optimizer_start.elapsed().as_secs_f64() * 1000.0;
        stats.diagnostics = updates.diagnostics;

        Ok(stats)
    }

    fn sync_loss(&mut self, batch: &TokenBatch, stats: &mut TrainStats) -> AppResult {
        let loss_sync_start = Instant::now();
        let losses = self
            .buffers
            .backward
            .losses
            .to_host_vec(self.runtime.stream.as_ref())?;
        let active_losses = &losses[..batch.token_count];
        stats.loss = active_losses.iter().sum::<f32>() / active_losses.len() as f32;
        stats.finite &= active_losses.iter().all(|value| value.is_finite());
        stats.nonzero |= active_losses.iter().any(|value| value.abs() > 0.0);
        stats.loss_host_wait_ms = loss_sync_start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}
