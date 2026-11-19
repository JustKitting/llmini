use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::optimizer::OptimizerModule;
use std::cell::Cell;

use super::tensor::{AdamState, StateInit, TokenEmbeddingState};
use crate::{
    training::{
        device_buffer::block_array,
        learning_rate::schedule_free_average_coefficient,
        update_skip::{UpdateSkipDecision, UpdateSkipState},
    },
    upload::UploadedModel,
};

mod components;

pub(in crate::training) use components::{BlockState, LayerNormState, LinearState, NextLatState};

pub struct OptimizerStateBuffers {
    pub(in crate::training) step: u32,
    pub(in crate::training) schedule_free_weight_sum: f32,
    pub(in crate::training) update_skip: UpdateSkipState,
    pub(in crate::training) token_embedding: TokenEmbeddingState,
    pub(in crate::training) xsa_alphas: AdamState,
    pub(in crate::training) ln_f: LayerNormState,
    pub(in crate::training) next_latent: NextLatState,
    pub(in crate::training) blocks: [BlockState; GPT2_N_LAYER],
    adam_schedule_amax_ptrs: DeviceBuffer<u64>,
    adam_schedule_amax_count: u32,
    training_weight_step: Cell<Option<u32>>,
    scheduled_values_valid: Cell<bool>,
}

impl OptimizerStateBuffers {
    pub fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        optimizer: &OptimizerModule,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        let init = StateInit::new(stream, decode, optimizer, super::super::symexp_lin::beta());
        let token_embedding = TokenEmbeddingState::new(init, &uploaded.token_embedding)?;
        let xsa_alphas = AdamState::new(init, &uploaded.xsa_alphas)?;
        let ln_f = LayerNormState::new(init, &uploaded.ln_f)?;
        let next_latent = NextLatState::new(init, &uploaded.next_latent)?;
        let blocks = block_array(|i| BlockState::new(init, &uploaded.blocks[i], i))?;

        let mut adam_schedule_amax_ptrs = Vec::new();
        token_embedding.push_schedule_amax_ptr(&mut adam_schedule_amax_ptrs);
        adam_schedule_amax_ptrs.push(xsa_alphas.schedule_amax_ptr());
        ln_f.push_schedule_amax_ptrs(&mut adam_schedule_amax_ptrs);
        next_latent.push_schedule_amax_ptrs(&mut adam_schedule_amax_ptrs);
        for block in &blocks {
            block.push_schedule_amax_ptrs(&mut adam_schedule_amax_ptrs);
        }
        let adam_schedule_amax_count = adam_schedule_amax_ptrs.len() as u32;
        let adam_schedule_amax_ptrs = DeviceBuffer::from_host(stream, &adam_schedule_amax_ptrs)?;

        Ok(Self {
            step: 0,
            schedule_free_weight_sum: 0.0,
            update_skip: UpdateSkipState::new(),
            token_embedding,
            xsa_alphas,
            ln_f,
            next_latent,
            blocks,
            adam_schedule_amax_ptrs,
            adam_schedule_amax_count,
            training_weight_step: Cell::new(None),
            scheduled_values_valid: Cell::new(false),
        })
    }

    pub(in crate::training) fn adam_schedule_amax_table(&self) -> (&DeviceBuffer<u64>, u32) {
        (&self.adam_schedule_amax_ptrs, self.adam_schedule_amax_count)
    }

    pub(in crate::training) fn advance(&mut self) -> u32 {
        self.step += 1;
        self.step
    }

    pub(in crate::training) fn schedule_free_average_coefficient(&mut self, step: u32) -> f32 {
        schedule_free_average_coefficient(step, &mut self.schedule_free_weight_sum)
    }

    pub(in crate::training) fn next_step(&self) -> u32 {
        self.step + 1
    }

    pub(in crate::training) fn training_weights_ready(&self) -> bool {
        self.training_weight_step.get() == Some(self.next_step())
    }

    pub(in crate::training) fn mark_training_weights_ready(&self) {
        self.training_weight_step.set(Some(self.next_step()));
    }

    pub(in crate::training) fn invalidate_training_weights(&self) {
        self.training_weight_step.set(None);
    }

    pub(in crate::training) fn scheduled_values_valid(&self) -> bool {
        self.scheduled_values_valid.get()
    }

    pub(in crate::training) fn invalidate_scheduled_values(&self) {
        self.scheduled_values_valid.set(false);
    }

    pub(in crate::training) fn mark_scheduled_values_ready(&self) {
        self.scheduled_values_valid.set(true);
    }

    pub(in crate::training) fn should_skip_update(
        &mut self,
        loss: Option<f32>,
        grad_norm: f32,
    ) -> UpdateSkipDecision {
        self.update_skip.observe(loss, grad_norm)
    }
}
