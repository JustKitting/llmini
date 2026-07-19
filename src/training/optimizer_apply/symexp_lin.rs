use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::{Fp32AdamWUpdateArgs, OptimizerModule};

use super::super::optimizer_state::{MuonState, OptimizerStateBuffers, SymExpLinScalarState};
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
    state: &mut OptimizerStateBuffers,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
) -> Result<(), DriverError> {
    if !super::super::symexp_lin::learn_scales() {
        return Ok(());
    }

    for block in &mut state.blocks {
        update_matrix(
            stream,
            optimizer,
            &mut block.attn_qkv.weight_muon,
            step,
            average_coefficient,
            grad_scale,
        )?;
        update_matrix(
            stream,
            optimizer,
            &mut block.attn_c_proj.weight_muon,
            step,
            average_coefficient,
            grad_scale,
        )?;
        update_matrix(
            stream,
            optimizer,
            &mut block.mlp_up.weight_muon,
            step,
            average_coefficient,
            grad_scale,
        )?;
        update_matrix(
            stream,
            optimizer,
            &mut block.mlp_down.weight_muon,
            step,
            average_coefficient,
            grad_scale,
        )?;
    }

    update_matrix(
        stream,
        optimizer,
        &mut state.next_latent.input_projection.weight_muon,
        step,
        average_coefficient,
        grad_scale,
    )?;
    update_matrix(
        stream,
        optimizer,
        &mut state.next_latent.transition.weight_muon,
        step,
        average_coefficient,
        grad_scale,
    )?;
    update_matrix(
        stream,
        optimizer,
        &mut state.next_latent.output_projection.weight_muon,
        step,
        average_coefficient,
        grad_scale,
    )
}

fn update_matrix(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    state: &mut MuonState,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
) -> Result<(), DriverError> {
    let anneal_steps = super::super::env::env_usize("TRAIN_SYMEXP_LIN_ANNEAL_STEPS")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(PAPER_WIDTH_2048_STEPS)
        .max(1);
    // The paper defines k relative to the optimizer's base LR. This repo applies
    // a separately tuned 12.5x multiplier to Adam-only model tensors, which must
    // not be multiplied by k a second time for the SEL controls.
    let base_lr = adam_learning_rate(step) / super::super::learning_rate::adam_scale();
    update_scalar(
        stream,
        optimizer,
        &mut state.symexp_lin.exponential,
        step,
        average_coefficient,
        grad_scale,
        base_lr
            * log_cosine_multiplier(step, anneal_steps, EXPONENTIAL_LR_START, EXPONENTIAL_LR_END),
        ADAM_WEIGHT_DECAY,
    )?;
    update_scalar(
        stream,
        optimizer,
        &mut state.symexp_lin.linear,
        step,
        average_coefficient,
        grad_scale,
        base_lr * log_cosine_multiplier(step, anneal_steps, LINEAR_LR_START, LINEAR_LR_END),
        ADAM_WEIGHT_DECAY,
    )?;
    update_scalar(
        stream,
        optimizer,
        &mut state.symexp_lin.curvature,
        step,
        average_coefficient,
        grad_scale,
        base_lr * log_cosine_multiplier(step, anneal_steps, CURVATURE_LR_START, CURVATURE_LR_END),
        0.0,
    )
}

#[expect(clippy::too_many_arguments)]
fn update_scalar(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    state: &mut SymExpLinScalarState,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<(), DriverError> {
    optimizer.apply_fp32_adamw_update(Fp32AdamWUpdateArgs {
        stream,
        z_master: &mut state.z_master,
        x_master: &mut state.x_master,
        grad: &state.grad,
        grad_scale,
        first_moment: &mut state.first,
        second_moment: &mut state.second,
        len: 1,
        learning_rate,
        weight_decay,
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        beta1_correction: 1.0 - ADAM_BETA1.powi(step as i32),
        beta2_correction: 1.0 - ADAM_BETA2.powi(step as i32),
        eps: ADAM_EPS,
        average_coefficient,
    })
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
