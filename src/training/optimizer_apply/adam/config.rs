const ADAM_LR: f32 = 2.0e-4;
pub(in crate::training::optimizer_apply) const ADAM_WEIGHT_DECAY: f32 = 0.005;
pub(in crate::training::optimizer_apply) const ADAM_BETA1: f32 = 0.9;
pub(in crate::training::optimizer_apply) const ADAM_BETA2: f32 = 0.95;
pub(in crate::training::optimizer_apply) const ADAM_EPS: f32 = 1.0e-10;

#[derive(Clone, Copy)]
pub(crate) struct AdamDebugConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta1_correction: f32,
    pub beta2_correction: f32,
    pub eps: f32,
}

pub(in crate::training::optimizer_apply) fn adam_learning_rate(step: u32) -> f32 {
    ADAM_LR * super::super::super::learning_rate::adam_multiplier(step)
}

pub(in crate::training::optimizer_apply) fn next_latent_adam_learning_rate(step: u32) -> f32 {
    ADAM_LR * super::super::super::learning_rate::next_latent_adam_multiplier(step)
}

pub(crate) fn adam_debug_config(step: u32) -> AdamDebugConfig {
    AdamDebugConfig {
        learning_rate: adam_learning_rate(step),
        weight_decay: ADAM_WEIGHT_DECAY,
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        beta1_correction: 1.0 - ADAM_BETA1.powi(step as i32),
        beta2_correction: 1.0 - ADAM_BETA2.powi(step as i32),
        eps: ADAM_EPS,
    }
}
