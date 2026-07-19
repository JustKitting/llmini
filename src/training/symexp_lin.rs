use cuda_core::{DeviceBuffer, DriverError};

use super::Trainer;
use super::grads::BackwardBuffers;
use super::next_latent::NextLatGradBuffers;
use super::optimizer_muon::{MuonPointerTables, apply_symexp_lin_chain_rule};
use super::optimizer_state::{AdamState, OptimizerStateBuffers};
use super::runtime::Runtime;
use crate::AppResult;

const DEFAULT_BETA: f32 = 12.5;

pub(super) fn enabled() -> bool {
    super::env::env_bool("TRAIN_SYMEXP_LIN").unwrap_or(true)
}

pub(super) fn beta() -> f32 {
    if !enabled() {
        return 0.0;
    }
    super::env::env_f32("TRAIN_SYMEXP_LIN_BETA")
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(DEFAULT_BETA)
}

pub(super) fn learn_scales() -> bool {
    enabled() && super::env::env_bool("TRAIN_SYMEXP_LIN_LEARN_SCALES").unwrap_or(true)
}

pub(super) fn reuse_precomputed_amax() -> bool {
    enabled() && super::env::env_bool("TRAIN_SYMEXP_LIN_REUSE_AMAX").unwrap_or(true)
}

pub(super) fn apply_gradient_chain_rule(
    runtime: &Runtime,
    tables: &MuonPointerTables,
    grads: &mut BackwardBuffers,
    next_latent_grads: &mut NextLatGradBuffers,
    state: &OptimizerStateBuffers,
    schedule_beta: f32,
) -> Result<(), DriverError> {
    let beta = beta();
    if beta == 0.0 {
        return Ok(());
    }

    apply_symexp_lin_chain_rule(runtime, tables, schedule_beta, beta, learn_scales())?;

    for (grad, state) in grads.blocks.iter_mut().zip(state.blocks.iter()) {
        chain_bias(
            runtime,
            &mut grad.d_attn_qkv_bias,
            &state.attn_qkv.bias,
            schedule_beta,
            beta,
        )?;
        chain_bias(
            runtime,
            &mut grad.d_attn_c_proj_bias,
            &state.attn_c_proj.bias,
            schedule_beta,
            beta,
        )?;
        chain_bias(
            runtime,
            &mut grad.d_mlp_c_fc_bias,
            &state.mlp_up.bias,
            schedule_beta,
            beta,
        )?;
        chain_bias(
            runtime,
            &mut grad.d_mlp_c_proj_bias,
            &state.mlp_down.bias,
            schedule_beta,
            beta,
        )?;
    }

    chain_bias(
        runtime,
        &mut next_latent_grads.d_input_projection_bias,
        &state.next_latent.input_projection.bias,
        schedule_beta,
        beta,
    )?;
    chain_bias(
        runtime,
        &mut next_latent_grads.d_transition_bias,
        &state.next_latent.transition.bias,
        schedule_beta,
        beta,
    )?;
    chain_bias(
        runtime,
        &mut next_latent_grads.d_output_projection_bias,
        &state.next_latent.output_projection.bias,
        schedule_beta,
        beta,
    )
}

fn chain_bias(
    runtime: &Runtime,
    grad: &mut DeviceBuffer<f32>,
    state: &AdamState,
    schedule_beta: f32,
    beta: f32,
) -> Result<(), DriverError> {
    runtime.optimizer.symexp_lin_chain_rule(
        runtime.stream.as_ref(),
        grad,
        &state.z_master,
        &state.x_master,
        schedule_beta,
        beta,
    )
}

pub(super) fn scale_summary(trainer: &Trainer) -> AppResult<Option<String>> {
    if !learn_scales() {
        return Ok(None);
    }
    let stream = trainer.runtime.stream.as_ref();
    let mut totals = [0.0_f64; 6];
    let mut count = 0usize;
    let mut add = |state: &super::optimizer_state::MuonState| -> AppResult {
        let values = [
            state.symexp_lin.exponential.z_master.to_host_vec(stream)?[0],
            state.symexp_lin.exponential.x_master.to_host_vec(stream)?[0],
            state.symexp_lin.linear.z_master.to_host_vec(stream)?[0],
            state.symexp_lin.linear.x_master.to_host_vec(stream)?[0],
            state.symexp_lin.curvature.z_master.to_host_vec(stream)?[0],
            state.symexp_lin.curvature.x_master.to_host_vec(stream)?[0],
        ];
        for (total, value) in totals.iter_mut().zip(values) {
            *total += value as f64;
        }
        count += 1;
        Ok(())
    };

    for block in &trainer.buffers.optimizer_state.blocks {
        add(&block.attn_qkv.weight_muon)?;
        add(&block.attn_c_proj.weight_muon)?;
        add(&block.mlp_up.weight_muon)?;
        add(&block.mlp_down.weight_muon)?;
    }
    add(&trainer
        .buffers
        .optimizer_state
        .next_latent
        .input_projection
        .weight_muon)?;
    add(&trainer
        .buffers
        .optimizer_state
        .next_latent
        .transition
        .weight_muon)?;
    add(&trainer
        .buffers
        .optimizer_state
        .next_latent
        .output_projection
        .weight_muon)?;

    let denominator = count as f64;
    Ok(Some(format!(
        "symexp_lin_scale_summary matrices={count} exponential_z_mean={:.6} exponential_x_mean={:.6} linear_z_mean={:.6} linear_x_mean={:.6} curvature_z_mean={:.6} curvature_x_mean={:.6}",
        totals[0] / denominator,
        totals[1] / denominator,
        totals[2] / denominator,
        totals[3] / denominator,
        totals[4] / denominator,
        totals[5] / denominator,
    )))
}

#[cfg(test)]
mod tests {
    fn congruent(raw: f32, beta: f32) -> f32 {
        raw.signum() * ((beta * raw.abs()).exp() - 1.0) / beta + raw / beta
    }

    fn mismatch(raw: f32, beta: f32) -> f32 {
        raw.signum() * (((beta * raw.abs()).exp() - 1.0) / beta + raw / beta)
    }

    fn inverse_congruent(target: f32, beta: f32) -> f32 {
        if target == 0.0 {
            return 0.0;
        }
        let mut magnitude = target.abs() / (1.0 + 1.0 / beta);
        for _ in 0..10 {
            let exponential = (beta * magnitude).exp();
            let residual = (exponential - 1.0 + magnitude) / beta - target.abs();
            magnitude = (magnitude - residual / (exponential + 1.0 / beta)).max(0.0);
        }
        target.signum() * magnitude
    }

    #[test]
    fn congruent_newton_inversion_recovers_targets() {
        for target in [-0.04_f32, -0.01, 0.0, 0.01, 0.04] {
            let raw = inverse_congruent(target, 12.5);
            assert!((congruent(raw, 12.5) - target).abs() < 2.0e-6);
        }
    }

    #[test]
    fn mismatch_preserves_positive_and_weakens_negative_initial_weights() {
        let beta = 12.5;
        let positive_target = 0.02;
        let negative_target = -0.02;
        let positive = mismatch(inverse_congruent(positive_target, beta), beta);
        let negative = mismatch(inverse_congruent(negative_target, beta), beta);
        assert!((positive - positive_target).abs() < 2.0e-6);
        assert!(negative < 0.0);
        assert!(negative.abs() < negative_target.abs());
    }

    #[test]
    fn mismatch_derivative_matches_finite_difference_off_zero() {
        let beta = 12.5;
        let epsilon = 1.0e-5;
        for raw in [-0.03_f32, -0.01, 0.01, 0.03] {
            let numerical =
                (mismatch(raw + epsilon, beta) - mismatch(raw - epsilon, beta)) / (2.0 * epsilon);
            let analytical = (beta * raw.abs()).exp() + raw.signum() / beta;
            assert!((numerical - analytical).abs() < 2.0e-3);
        }
    }
}
