use cuda_core::DeviceBuffer;
use gpt2_nvfp4::Gpt2Rng;
use rust_kernels_cuda::hnet_stage::RatioObjective;

use super::{Trainer, bytes::ByteBatch, hierarchy::HierarchyScratch, parameters::Parameters};
use crate::{
    AppResult,
    training::{
        grad_clip::GradientClipBuffers,
        learning_rate::schedule_free_beta,
        optimizer_apply::{WeightUpdateArgs, apply_backbone_weight_updates},
        optimizer_muon::MuonPointerTables,
    },
};

pub(super) struct HnetStep {
    pub nll_sum: f32,
    pub ratio_loss: f32,
    pub grad_norm: f32,
    pub skipped: bool,
}

impl Trainer {
    pub(super) fn train_hnet_step(
        &mut self,
        scratch: &mut HierarchyScratch,
        parameters: &mut Parameters,
        batch: &ByteBatch,
        ratio: RatioObjective,
        rng: &mut Gpt2Rng,
    ) -> AppResult<HnetStep> {
        let step = self.buffers.optimizer_state.next_step();
        parameters.materialize(
            &self.runtime,
            &mut self.buffers.optimizer,
            schedule_free_beta(step),
        )?;
        let weights = parameters.weights();
        let (adjoints, lookup) = self.backward_hnet_bytes(
            scratch,
            batch,
            weights.embedding,
            weights.hierarchy(),
            ratio,
            rng,
        )?;
        let stream = &self.runtime.stream;
        parameters.bind_gradients(stream, &adjoints, &lookup)?;
        let nll_sum: f32 = adjoints.byte_losses.to_host_vec(stream)?.into_iter().sum();
        let ratio_loss = adjoints.ratio_loss.to_host_vec(stream)?[0];
        let loss = nll_sum / batch.scored_targets as f32 + ratio_loss;
        let result = self.update_hnet_backbone(adjoints.backbone_rows, Some(loss))?;
        parameters.update(&self.runtime, step, &result, rng)?;
        Ok(HnetStep {
            nll_sum,
            ratio_loss,
            grad_norm: result.trace.grad_norm,
            skipped: result.trace.update_skipped,
        })
    }

    /// Install the routed-backbone update tables and include the hierarchy's
    /// parameter gradients in the same norm reduction. These buffers must
    /// keep their device addresses for the lifetime of the training loop.
    pub(super) fn prepare_hnet_optimizer(
        &mut self,
        hierarchy_gradients: &[(&str, &DeviceBuffer<f32>)],
    ) -> AppResult {
        self.ensure_ready()?;
        if self.engram.is_some() || self.latent_feedback.is_some() {
            return Err("token-indexed modules do not have a routed-backbone update".into());
        }
        let stream = &self.runtime.stream;
        let tables = MuonPointerTables::new_backbone(
            stream,
            &self.uploaded,
            &self.buffers.backward,
            &self.buffers.optimizer_state,
            self.mhc.as_ref(),
        )?;
        let mut extra = hierarchy_gradients.to_vec();
        let recompute = self.mhc.as_ref().map_or_else(Vec::new, |mhc| {
            extra.extend(mhc.named_gradients());
            mhc.units
                .iter()
                .map(|unit| unit.grad.cu_deviceptr())
                .collect()
        });
        let clip = GradientClipBuffers::new_backbone(
            stream,
            &self.buffers.backward,
            &tables,
            &extra,
            &recompute,
        )?;
        self.buffers.muon_tables = tables;
        self.buffers.grad_clip = clip;
        self.buffers.optimizer_state.invalidate_scheduled_values();
        self.buffers.optimizer_state.invalidate_training_weights();
        Ok(())
    }

    pub(super) fn update_hnet_backbone(
        &mut self,
        rows: u32,
        observed_loss: Option<f32>,
    ) -> AppResult<crate::training::optimizer_apply::WeightUpdateResult> {
        let updates = apply_backbone_weight_updates(WeightUpdateArgs {
            stream: &self.runtime.stream,
            runtime: &self.runtime,
            batch: rows,
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
        if !updates.trace.update_skipped
            && let Some(mhc) = &mut self.mhc
        {
            mhc.update_vectors(
                &self.runtime,
                self.buffers.optimizer_state.step,
                updates.average_coefficient,
                updates.grad_scale,
            )?;
        }
        Ok(updates)
    }
}
