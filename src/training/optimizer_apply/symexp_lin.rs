use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use super::super::optimizer_muon::MuonPointerTables;
use super::adam::{ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WEIGHT_DECAY, adam_learning_rate};

const EXPONENTIAL_LR_START: f32 = 50.0;
const EXPONENTIAL_LR_END: f32 = 8.0;
const LINEAR_LR_START: f32 = 50.0;
const LINEAR_LR_END: f32 = 8.0;
const CURVATURE_LR_START: f32 = 0.01;
const CURVATURE_LR_END: f32 = 0.5;
const PAPER_WIDTH_2048_STEPS: u32 = 150_000;

pub(super) fn update_global_scales(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    tables: &MuonPointerTables,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
) -> Result<(), DriverError> {
    if !super::super::symexp_lin::learn_scales() {
        return Ok(());
    }

    let anneal_steps = super::super::env::env_usize("TRAIN_SYMEXP_LIN_ANNEAL_STEPS")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(PAPER_WIDTH_2048_STEPS)
        .max(1);
    // The paper defines k relative to the optimizer's base LR. This repo applies
    // a separately tuned 12.5x multiplier to Adam-only model tensors, which must
    // not be multiplied by k a second time for the SEL controls.
    let base_lr = adam_learning_rate(step) / super::super::learning_rate::adam_scale();
    let exponential_linear_learning_rate = base_lr
        * log_cosine_multiplier(step, anneal_steps, EXPONENTIAL_LR_START, EXPONENTIAL_LR_END);
    debug_assert_eq!(
        exponential_linear_learning_rate,
        base_lr * log_cosine_multiplier(step, anneal_steps, LINEAR_LR_START, LINEAR_LR_END)
    );
    let curvature_learning_rate =
        base_lr * log_cosine_multiplier(step, anneal_steps, CURVATURE_LR_START, CURVATURE_LR_END);
    optimizer.symexp_lin_scale_adamw_update(
        stream,
        tables.symexp_lin_slots(),
        tables.slot_count as u32,
        grad_scale,
        exponential_linear_learning_rate,
        curvature_learning_rate,
        ADAM_WEIGHT_DECAY,
        ADAM_BETA1,
        ADAM_BETA2,
        1.0 - ADAM_BETA1.powi(step as i32),
        1.0 - ADAM_BETA2.powi(step as i32),
        ADAM_EPS,
        average_coefficient,
    )
}

fn log_cosine_multiplier(step: u32, anneal_steps: u32, start: f32, end: f32) -> f32 {
    let progress = (step as f32 / anneal_steps as f32).min(1.0);
    let blend = 0.5 * (1.0 - (core::f32::consts::PI * progress).cos());
    (start.ln() + (end / start).ln() * blend).exp()
}

#[cfg(test)]
mod tests {
    use super::log_cosine_multiplier;

    #[test]
    fn paper_scale_schedule_hits_both_endpoints() {
        let start = log_cosine_multiplier(0, 100, 50.0, 8.0);
        let end = log_cosine_multiplier(100, 100, 50.0, 8.0);
        assert!((start - 50.0).abs() < 1.0e-5);
        assert!((end - 8.0).abs() < 1.0e-5);
    }
}
