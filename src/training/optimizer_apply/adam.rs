mod config;

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{AdamWUpdateArgs, OptimizerModule};

use crate::upload::UploadedNvfp4;

pub(crate) use config::adam_debug_config;
pub(super) use config::{
    ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY, adam_learning_rate,
    next_latent_adam_learning_rate,
};

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::AdamState;
use super::timed_ms;

pub(super) struct AdamUpdate<'a, 'scratch> {
    stream: &'a CudaStream,
    optimizer: &'a OptimizerModule,
    scratch: &'scratch mut OptimizerScratch,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
    learning_rate: f32,
}

impl<'a, 'scratch> AdamUpdate<'a, 'scratch> {
    pub(super) fn new(
        stream: &'a CudaStream,
        optimizer: &'a OptimizerModule,
        scratch: &'scratch mut OptimizerScratch,
        step: u32,
        average_coefficient: f32,
        grad_scale: f32,
    ) -> Self {
        Self::with_learning_rate(
            stream,
            optimizer,
            scratch,
            step,
            average_coefficient,
            grad_scale,
            adam_learning_rate(step),
        )
    }

    pub(super) fn with_learning_rate(
        stream: &'a CudaStream,
        optimizer: &'a OptimizerModule,
        scratch: &'scratch mut OptimizerScratch,
        step: u32,
        average_coefficient: f32,
        grad_scale: f32,
        learning_rate: f32,
    ) -> Self {
        Self {
            stream,
            optimizer,
            scratch,
            step,
            average_coefficient,
            grad_scale,
            learning_rate,
        }
    }

    pub(super) fn update(
        &mut self,
        tensor: &mut UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &mut AdamState,
    ) -> Result<(), DriverError> {
        self.optimizer
            .apply_adamw_update_deferred_quantization(AdamWUpdateArgs {
                stream: self.stream,
                bytes: &mut tensor.bytes,
                scales: &mut tensor.scales,
                global_scale: &mut tensor.global_scale,
                z_master: &mut state.z_master,
                x_master: &mut state.x_master,
                grad,
                grad_scale: self.grad_scale,
                first_moment: &mut state.first,
                second_moment: &mut state.second,
                amax: &mut self.scratch.amax,
                chunk_amax: &mut self.scratch.chunk_amax,
                len: tensor.len as u32,
                learning_rate: self.learning_rate,
                weight_decay: ADAM_WEIGHT_DECAY,
                beta1: ADAM_BETA1,
                beta2: ADAM_BETA2,
                beta1_correction: 1.0 - ADAM_BETA1.powi(self.step as i32),
                beta2_correction: 1.0 - ADAM_BETA2.powi(self.step as i32),
                eps: ADAM_EPS,
                average_coefficient: self.average_coefficient,
            })
    }

    pub(super) fn update_timed(
        &mut self,
        tensor: &mut UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &mut AdamState,
    ) -> Result<f64, DriverError> {
        timed_ms(|| self.update(tensor, grad, state))
    }
}
