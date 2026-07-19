use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_VOCAB_SIZE};
use rust_kernels_cuda::optimizer::{EmberUpdateArgs, OptimizerModule};

use crate::upload::UploadedNvfp4;

use super::super::learning_rate::warmup_multiplier;
use super::super::optimizer_state::EmberState;
use super::timed_ms;

const DEFAULT_EMBER_LR: f32 = 1.0e-3;
const DEFAULT_EMBER_BETA2: f32 = 0.999;
const DEFAULT_EMBER_WEIGHT_DECAY: f32 = 0.0;
const DEFAULT_EMBER_EPS: f32 = 1.0e-8;

pub(in crate::training) fn enabled() -> bool {
    super::super::env::env_bool("TRAIN_EMBER").unwrap_or(true)
}

pub(in crate::training) fn learning_rate(step: u32) -> f32 {
    configured_positive("TRAIN_EMBER_LR", DEFAULT_EMBER_LR) * warmup_multiplier(step)
}

pub(in crate::training) fn beta2() -> f32 {
    super::super::env::env_f32("TRAIN_EMBER_BETA2")
        .filter(|value| value.is_finite())
        .unwrap_or(DEFAULT_EMBER_BETA2)
        .clamp(0.0, 0.999_999)
}

pub(in crate::training) fn weight_decay() -> f32 {
    configured_nonnegative("TRAIN_EMBER_WEIGHT_DECAY", DEFAULT_EMBER_WEIGHT_DECAY)
}

pub(in crate::training) fn eps() -> f32 {
    configured_positive("TRAIN_EMBER_EPS", DEFAULT_EMBER_EPS)
}

fn configured_positive(name: &str, default: f32) -> f32 {
    super::super::env::env_f32(name)
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}

fn configured_nonnegative(name: &str, default: f32) -> f32 {
    super::super::env::env_f32(name)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

pub(super) struct EmberUpdate<'a> {
    stream: &'a CudaStream,
    optimizer: &'a OptimizerModule,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
}

impl<'a> EmberUpdate<'a> {
    pub(super) fn new(
        stream: &'a CudaStream,
        optimizer: &'a OptimizerModule,
        step: u32,
        average_coefficient: f32,
        grad_scale: f32,
    ) -> Self {
        Self {
            stream,
            optimizer,
            step,
            average_coefficient,
            grad_scale,
        }
    }

    fn update(
        &self,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &mut EmberState,
    ) -> Result<(), DriverError> {
        let beta2 = beta2();
        let rows = GPT2_VOCAB_SIZE as u32;
        let cols = GPT2_N_EMBD as u32;
        assert_eq!(tensor.len, rows as usize * cols as usize);
        self.optimizer.apply_ember_update(EmberUpdateArgs {
            stream: self.stream,
            z_master: &mut state.z_master,
            x_master: &mut state.x_master,
            grad,
            row_second_moment: &mut state.row_second_moment,
            column_second_moment: &mut state.column_second_moment,
            column_partials: &mut state.column_partials,
            normalizer: &mut state.normalizer,
            rows,
            cols,
            grad_scale: self.grad_scale,
            learning_rate: learning_rate(self.step),
            weight_decay: weight_decay(),
            beta2,
            beta2_correction: 1.0 - beta2.powi(self.step as i32),
            eps: eps(),
            average_coefficient: self.average_coefficient,
        })
    }

    pub(super) fn update_timed(
        &self,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &mut EmberState,
    ) -> Result<f64, DriverError> {
        timed_ms(|| self.update(tensor, grad, state))
    }
}
