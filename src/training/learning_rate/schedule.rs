use super::config::{
    adam_scale, amuse_beta1, amuse_rho, next_latent_scale, scale, start_ratio, warmup_steps,
};

const AMUSE_AVERAGE_R: f32 = 0.0;
const AMUSE_WEIGHT_LR_POWER: f32 = 2.0;

pub(in crate::training) fn adam_multiplier(step: u32) -> f32 {
    adam_scale() * warmup_only(step)
}

pub(in crate::training) fn next_latent_adam_multiplier(step: u32) -> f32 {
    adam_multiplier(step) * next_latent_scale()
}

pub(in crate::training) fn muon_multiplier(step: u32) -> f32 {
    scale() * warmup_only(step)
}

pub(in crate::training) fn warmup_multiplier(step: u32) -> f32 {
    warmup_only(step)
}

pub(in crate::training) fn schedule_free_beta(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps().max(2);
    let beta1 = amuse_beta1();
    if step <= warmup {
        return beta1;
    }

    let ratio = (warmup - 1) as f32 / (step - 1) as f32;
    1.0 - ratio.powf(amuse_rho()) * (1.0 - beta1)
}

pub(in crate::training) fn schedule_free_average_coefficient(
    step: u32,
    weight_sum: &mut f32,
) -> f32 {
    let weight = schedule_free_average_weight(step);
    *weight_sum += weight;
    if *weight_sum > 0.0 {
        weight / *weight_sum
    } else {
        0.0
    }
}

fn warmup_only(step: u32) -> f32 {
    let step = step.max(1);
    let warmup = warmup_steps();
    if step <= warmup {
        let progress = step as f32 / warmup as f32;
        let start = start_ratio();
        start + (1.0 - start) * progress
    } else {
        1.0
    }
}

fn schedule_free_average_weight(step: u32) -> f32 {
    let step = step.max(1);
    let lr_max = warmup_only(step);
    (step as f32).powf(AMUSE_AVERAGE_R) * lr_max.powf(AMUSE_WEIGHT_LR_POWER)
}
