use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::GPT2_N_LAYER;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::optimizer::OptimizerModule;

use super::tensor::{AdamState, StateInit};
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
    pub(in crate::training) token_embedding: AdamState,
    pub(in crate::training) xsa_alphas: AdamState,
    pub(in crate::training) ln_f: LayerNormState,
    pub(in crate::training) next_latent: NextLatState,
    pub(in crate::training) blocks: [BlockState; GPT2_N_LAYER],
}

impl OptimizerStateBuffers {
    pub fn new(
        stream: &CudaStream,
        decode: &Nvfp4DecodeModule,
        optimizer: &OptimizerModule,
        uploaded: &UploadedModel,
    ) -> Result<Self, DriverError> {
        let init = StateInit::new(stream, decode, optimizer, super::super::symexp_lin::beta());
        Ok(Self {
            step: 0,
            schedule_free_weight_sum: 0.0,
            update_skip: UpdateSkipState::new(),
            token_embedding: AdamState::new(init, &uploaded.token_embedding)?,
            xsa_alphas: AdamState::new(init, &uploaded.xsa_alphas)?,
            ln_f: LayerNormState::new(init, &uploaded.ln_f)?,
            next_latent: NextLatState::new(init, &uploaded.next_latent)?,
            blocks: block_array(|i| BlockState::new(init, &uploaded.blocks[i]))?,
        })
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

    pub(in crate::training) fn should_skip_update(
        &mut self,
        loss: Option<f32>,
        grad_norm: f32,
    ) -> UpdateSkipDecision {
        self.update_skip.observe(loss, grad_norm)
    }
}
