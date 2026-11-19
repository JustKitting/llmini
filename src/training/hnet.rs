//! Run the native backbone on H-Net chunks and keep its training tape live.
use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_MLP_ROUTE_TILE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use rust_kernels_cuda::hnet_stage::{HnetStage, StagePlan, WidthAdjoints};

use super::{Trainer, forward::ForwardInput, learning_rate::schedule_free_beta, schedule_free};
use crate::AppResult;

mod bytes;
mod hierarchy;
mod optimizer;
mod outer;
mod parameters;
mod projection;

pub(in crate::training) struct RoutedBackboneTape<'trainer, 'stage, 'plan> {
    trainer: &'trainer mut Trainer,
    stage: &'stage HnetStage,
    plan: &'stage StagePlan<'plan>,
    output: DeviceBuffer<u16>,
}

impl Trainer {
    pub(in crate::training) fn forward_routed_backbone<'trainer, 'stage, 'plan>(
        &'trainer mut self,
        stage: &'stage HnetStage,
        plan: &'stage StagePlan<'plan>,
        encoded: &DeviceBuffer<u16>,
        learned_padding: Option<&DeviceBuffer<u16>>,
    ) -> AppResult<RoutedBackboneTape<'trainer, 'stage, 'plan>> {
        self.ensure_ready()?;
        if self.engram.is_some() || self.latent_feedback.is_some() {
            return Err(
                "a routed backbone has neither token-indexed Engram nor token feedback".into(),
            );
        }
        let batch_size = plan.batch();
        let seq_len = plan.inner_stride();
        if batch_size as usize > GPT2_BATCH_SIZE
            || seq_len as usize > GPT2_SEQ_LEN
            || plan.features() as usize > GPT2_N_EMBD
            || !seq_len.is_multiple_of(GPT2_MLP_ROUTE_TILE as u32)
            || !plan.uses_stream(&self.runtime.stream)
        {
            return Err("routed chunks do not fit the native backbone shape or stream".into());
        }

        schedule_free::materialize_backbone_training_weights(
            &self.runtime,
            &mut self.uploaded,
            &mut self.buffers.optimizer,
            &self.buffers.optimizer_state,
            &self.buffers.backward,
        )?;
        if let (Some(mhc), Some(weights)) = (&mut self.mhc, &mut self.uploaded.mhc) {
            mhc.materialize(
                &self.runtime,
                weights,
                &mut self.buffers.optimizer,
                schedule_free_beta(self.buffers.optimizer_state.next_step()),
                false,
            )?;
        }
        let chunks = stage.gather(plan, encoded)?;
        let inner_width = GPT2_N_EMBD as u32;
        stage.write_residual(
            plan,
            &chunks,
            inner_width,
            learned_padding,
            &mut self.buffers.residual,
        )?;
        self.forward_pass(
            ForwardInput::Residual {
                batch_size,
                seq_len,
            },
            false,
            false,
            false,
        )?;
        let output = stage.read_residual(plan, &self.buffers.residual, inner_width)?;
        Ok(RoutedBackboneTape {
            trainer: self,
            stage,
            plan,
            output,
        })
    }
}

#[cfg(test)]
mod tests;

impl RoutedBackboneTape<'_, '_, '_> {
    pub(in crate::training) fn output(&self) -> &DeviceBuffer<u16> {
        &self.output
    }

    /// Return the compact adjoint in the exact shape of the original gather.
    /// The caller scatters it and adds the encoder residual and router paths.
    pub(in crate::training) fn backward(
        self,
        output_gradient: &DeviceBuffer<f32>,
    ) -> AppResult<WidthAdjoints> {
        let inner_width = GPT2_N_EMBD as u32;
        let output_gradient =
            self.stage
                .widen_output_adjoint(self.plan, output_gradient, inner_width)?;
        self.trainer.enqueue_residual_backward(
            self.plan.batch(),
            self.plan.inner_stride(),
            &output_gradient,
        )?;
        Ok(self.stage.split_input_adjoint(
            self.plan,
            &self.trainer.buffers.backward.d_embedding_residual,
            inner_width,
        )?)
    }
}
