use cuda_core::CudaStream;

use crate::AppResult;
use crate::upload::UploadedModel;

use super::types::TrainingDiagnostics;
use super::update::{
    PendingTensorUpdateDiagnostics, changed_bytes, collect_update_snapshots,
    finish_update_snapshots,
};
use super::util::{f32_buffer_stats, f32_buffer_stats_scaled, hash_bytes};
use crate::training::grads::BackwardBuffers;
use crate::training::optimizer_state::OptimizerStateBuffers;

pub struct PendingTrainingDiagnostics {
    diagnostics: TrainingDiagnostics,
    token_embedding_bytes_before: Vec<u8>,
    updates: Vec<PendingTensorUpdateDiagnostics>,
}

impl PendingTrainingDiagnostics {
    pub fn collect(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        state: &OptimizerStateBuffers,
        step: u32,
        average_coefficient: f32,
        grad_scale: f32,
    ) -> AppResult<Self> {
        let token_embedding_bytes_before = uploaded.token_embedding.bytes.to_host_vec(stream)?;
        let (dlogits_rms, dlogits_max) = f32_buffer_stats(stream, &grads.dlogits)?;
        let (d_lm_head_rms, d_lm_head_max) =
            f32_buffer_stats_scaled(stream, &grads.d_lm_head_weight, grad_scale)?;
        let (d_embedding_rms, d_embedding_max) =
            f32_buffer_stats(stream, &grads.d_embedding_residual)?;
        let updates = collect_update_snapshots(
            stream,
            uploaded,
            grads,
            state,
            step,
            average_coefficient,
            grad_scale,
        )?;
        let token_embedding_global = uploaded.token_embedding.global_scale_to_host(stream)?;

        Ok(Self {
            diagnostics: TrainingDiagnostics {
                update_count: updates.len(),
                positive_update_dot_count: 0,
                zero_grad_changed_count: 0,
                max_update_to_weight_rms: 0.0,
                dlogits_rms,
                dlogits_max,
                d_lm_head_rms,
                d_lm_head_max,
                d_embedding_rms,
                d_embedding_max,
                token_embedding_global_before: token_embedding_global,
                token_embedding_global_after: token_embedding_global,
                token_embedding_changed_bytes: 0,
                token_embedding_hash_before: hash_bytes(&token_embedding_bytes_before),
                token_embedding_hash_after: 0,
                updates: Vec::new(),
            },
            token_embedding_bytes_before,
            updates,
        })
    }

    pub fn finish(
        mut self,
        stream: &CudaStream,
        uploaded: &UploadedModel,
    ) -> AppResult<TrainingDiagnostics> {
        let after = uploaded.token_embedding.bytes.to_host_vec(stream)?;
        self.diagnostics.token_embedding_global_after =
            uploaded.token_embedding.global_scale_to_host(stream)?;
        self.diagnostics.token_embedding_changed_bytes =
            changed_bytes(&self.token_embedding_bytes_before, &after);
        self.diagnostics.token_embedding_hash_after = hash_bytes(&after);
        let updates = finish_update_snapshots(stream, uploaded, self.updates)?;
        self.diagnostics.positive_update_dot_count = updates
            .iter()
            .filter(|update| update.delta_grad_dot > 0.0)
            .count();
        self.diagnostics.zero_grad_changed_count = updates
            .iter()
            .filter(|update| update.grad_nonzero == 0 && update.changed_bytes > 0)
            .count();
        self.diagnostics.max_update_to_weight_rms = updates
            .iter()
            .map(|update| update.update_to_weight_rms)
            .fold(0.0, f32::max);
        self.diagnostics.updates = updates;
        Ok(self.diagnostics)
    }
}
