use std::time::Instant;

mod pass;

use super::optimizer_apply::{WeightUpdateArgs, apply_weight_updates};
use super::{TokenBatch, TrainStats, Trainer};
use crate::AppResult;

impl Trainer {
    pub(in crate::training) fn enqueue_numeric_probe_backward(
        &mut self,
        batch: &TokenBatch,
    ) -> AppResult {
        self.ensure_ready()?;
        self.enqueue_backward(batch)
    }

    pub(in crate::training) fn materialize_training_weights(&mut self) -> AppResult<()> {
        self.ensure_ready()?;
        super::schedule_free::materialize_training_weights(
            self.runtime.stream.as_ref(),
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
            &self.buffers.backward,
            &self.buffers.next_latent_grads,
        )?;
        if let (Some(e), Some(w)) = (&mut self.engram, &mut self.uploaded.engram) {
            e.materialize_training(
                &self.runtime,
                w,
                &mut self.buffers.optimizer,
                super::learning_rate::schedule_free_beta(self.buffers.optimizer_state.next_step()),
            )?;
        }

        if let (Some(f), Some(w)) = (
            &mut self.latent_feedback,
            &mut self.uploaded.latent_feedback,
        ) {
            f.materialize_training(
                &self.runtime,
                w,
                &mut self.buffers.optimizer,
                super::learning_rate::schedule_free_beta(self.buffers.optimizer_state.next_step()),
            )?;
        }
        if let (Some(m), Some(w)) = (&mut self.mhc, &mut self.uploaded.mhc) {
            m.materialize(
                &self.runtime,
                w,
                &mut self.buffers.optimizer,
                super::learning_rate::schedule_free_beta(self.buffers.optimizer_state.next_step()),
                false,
            )?;
        }
        Ok(())
    }

    pub(in crate::training) fn materialize_evaluation_weights(&mut self) -> AppResult<()> {
        self.ensure_ready()?;
        super::schedule_free::materialize_evaluation_weights(
            self.runtime.stream.as_ref(),
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
        )?;
        if let (Some(e), Some(w)) = (&mut self.engram, &mut self.uploaded.engram) {
            e.materialize_evaluation(&self.runtime, w, &mut self.buffers.optimizer)?;
        }
        if let (Some(f), Some(w)) = (
            &mut self.latent_feedback,
            &mut self.uploaded.latent_feedback,
        ) {
            f.materialize_evaluation(&self.runtime, w, &mut self.buffers.optimizer)?;
        }
        if let (Some(m), Some(w)) = (&mut self.mhc, &mut self.uploaded.mhc) {
            m.materialize(&self.runtime, w, &mut self.buffers.optimizer, 0., true)?;
        }
        Ok(())
    }

    pub fn train_step(&mut self, batch: &TokenBatch, sync_loss: bool) -> AppResult<TrainStats> {
        let passes = if self.latent_feedback.is_some() {
            super::latent_feedback::training_passes(
                self.base_seed,
                self.buffers.optimizer_state.step,
            )
        } else {
            1
        };
        self.train_step_with_passes(batch, sync_loss, passes)
    }

    pub(in crate::training) fn train_step_with_passes(
        &mut self,
        batch: &TokenBatch,
        sync_loss: bool,
        passes: usize,
    ) -> AppResult<TrainStats> {
        self.materialize_training_weights()?;

        let forward_start = Instant::now();
        let mut stats = self.forward_for_passes(batch, passes)?;
        stats.forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

        let backward_start = Instant::now();
        self.enqueue_backward(batch)?;
        stats.backward_enqueue_ms = backward_start.elapsed().as_secs_f64() * 1000.0;

        if sync_loss {
            self.sync_loss(batch, &mut stats)?;
        }
        let observed_loss = sync_loss.then_some(stats.loss);

        #[cfg(test)]
        if let Some(mut audit) = self.update_rejection_audit.take() {
            audit.before_update(self)?;
            self.update_rejection_audit = Some(audit);
        }

        let optimizer_start = Instant::now();
        let stream = self.runtime.stream.as_ref();
        let updates = apply_weight_updates(WeightUpdateArgs {
            stream,
            runtime: &self.runtime,
            batch,
            uploaded: &mut self.uploaded,
            grads: &mut self.buffers.backward,
            next_latent_grads: &mut self.buffers.next_latent_grads,
            observed_loss,
            scratch: &mut self.buffers.optimizer,
            state: &mut self.buffers.optimizer_state,
            muon: &mut self.buffers.muon,
            muon_tables: &self.buffers.muon_tables,
            tape: &self.buffers.tape,
            qk_norm_max: self.buffers.scratch.attention_core.qk_norm_max(),
            grad_clip: &mut self.buffers.grad_clip,
        })?;
        if !updates.trace.update_skipped {
            if let Some(m) = &mut self.mhc {
                m.update_vectors(
                    &self.runtime,
                    self.buffers.optimizer_state.step,
                    updates.average_coefficient,
                    updates.grad_scale,
                )?;
            }
            if let (Some(e), Some(w)) = (&mut self.engram, &mut self.uploaded.engram) {
                e.update(
                    &self.runtime,
                    w,
                    &mut self.buffers.optimizer,
                    self.buffers.optimizer_state.step,
                    updates.average_coefficient,
                    updates.grad_scale,
                )?;
            }
        }
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
