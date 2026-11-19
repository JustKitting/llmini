//! Run the native backbone on H-Net chunks and keep its training tape live.
use cuda_core::{DeviceBuffer, memory};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_MLP_ROUTE_TILE, GPT2_N_EMBD, GPT2_SEQ_LEN};
use rust_kernels_cuda::hnet_stage::{HnetStage, StagePlan};

use super::{Trainer, forward::ForwardInput};
use crate::AppResult;

pub(in crate::training) struct RoutedBackboneTape<'a> {
    trainer: &'a mut Trainer,
    batch_size: u32,
    seq_len: u32,
    output: DeviceBuffer<u16>,
}

impl Trainer {
    pub(in crate::training) fn forward_routed_backbone<'a>(
        &'a mut self,
        stage: &HnetStage,
        plan: &StagePlan<'_>,
        encoded: &DeviceBuffer<u16>,
    ) -> AppResult<RoutedBackboneTape<'a>> {
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
            || plan.features() as usize != GPT2_N_EMBD
            || !seq_len.is_multiple_of(GPT2_MLP_ROUTE_TILE as u32)
            || !plan.uses_stream(&self.runtime.stream)
        {
            return Err("routed chunks do not fit the native backbone shape or stream".into());
        }

        self.materialize_training_weights()?;
        let chunks = stage.gather(plan, encoded)?;
        stage.write_residual(plan, &chunks, &mut self.buffers.residual)?;
        self.forward_pass(
            ForwardInput::Residual {
                batch_size,
                seq_len,
            },
            false,
            false,
            false,
        )?;
        let output = stage.read_residual(plan, &self.buffers.residual)?;
        Ok(RoutedBackboneTape {
            trainer: self,
            batch_size,
            seq_len,
            output,
        })
    }
}

#[cfg(test)]
mod tests;

impl RoutedBackboneTape<'_> {
    pub(in crate::training) fn output(&self) -> &DeviceBuffer<u16> {
        &self.output
    }

    /// Return the compact adjoint in the exact shape of the original gather.
    /// The caller scatters it and adds the encoder residual and router paths.
    pub(in crate::training) fn backward(
        self,
        output_gradient: &DeviceBuffer<f32>,
    ) -> AppResult<DeviceBuffer<f32>> {
        let values = self.output.len();
        if output_gradient.len() != values
            || !std::sync::Arc::ptr_eq(
                output_gradient.context(),
                self.trainer.runtime.stream.context(),
            )
        {
            return Err("the routed output gradient does not match the saved backbone tape".into());
        }
        self.trainer
            .enqueue_residual_backward(self.batch_size, self.seq_len, output_gradient)?;
        let stream = self.trainer.runtime.stream.as_ref();
        let compact = DeviceBuffer::zeroed(stream, values)?;
        let source = &self.trainer.buffers.backward.d_embedding_residual;
        // The allocated native buffer holds at least this compact prefix.
        unsafe {
            memory::memcpy_dtod_async(
                compact.cu_deviceptr(),
                source.cu_deviceptr(),
                compact.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        Ok(compact)
    }
}
