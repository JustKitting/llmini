//! Host-side Muon update flow.

mod groups;
mod tma;

use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

pub(super) use groups::{MuonGroupTable, MuonPointerTables};
pub(super) use tma::{MuonTmaArgs, apply_muon_tma};

const SIGN_MUON_BETA: f32 = 0.9;
const SIGN_MUON_PERIOD: u32 = 2;
const SIGN_MUON_FULL_LR_MULTIPLIER: f32 = 2.0;
const SIGN_MUON_SIGN_LR_RATIO: f32 = 0.025;
const HYPERBALL_LR: f32 = 0.022;
const HYPERBALL_MOMENTUM: f32 = 0.95;
const HYPERBALL_POLAR_PERIOD: usize = 3;
const HYPERBALL_POLAR_NUMERATOR: usize = 2;
const POLAR_ITERATIONS: u32 = 5;
pub(super) const MUON_LR: f32 = 1.0e-4;
pub(super) const MUON_WEIGHT_DECAY: f32 = 0.025;
pub(in crate::training) const MUON_MATRIX_SLOTS: usize = GPT2_N_LAYER * 4 + 3;

pub(super) fn hyperball_enabled() -> bool {
    super::env::env_bool("TRAIN_HYPERBALL").unwrap_or(true)
}

pub(super) fn hyperball_learning_rate() -> f32 {
    super::env::env_f32("TRAIN_HYPERBALL_LR")
        .unwrap_or(HYPERBALL_LR)
        .max(0.0)
}

pub(super) fn hyperball_momentum() -> f32 {
    HYPERBALL_MOMENTUM
}

pub(super) fn muon_vs_enabled() -> bool {
    super::env::env_bool("TRAIN_MUON_VS").unwrap_or(true)
}

pub(super) fn hyperball_uses_schedule_free() -> bool {
    super::env::env_bool("TRAIN_HYPERBALL_AMUSE").unwrap_or(true)
}

pub(super) fn hyperball_polar_period() -> usize {
    super::env::env_usize("TRAIN_HYPERBALL_POLAR_PERIOD")
        .unwrap_or(HYPERBALL_POLAR_PERIOD)
        .max(1)
}

pub(super) fn hyperball_polar_numerator() -> usize {
    let period = hyperball_polar_period();
    super::env::env_usize("TRAIN_HYPERBALL_POLAR_NUMERATOR")
        .unwrap_or(HYPERBALL_POLAR_NUMERATOR)
        .clamp(1, period)
}

pub(super) fn hyperball_uses_polar(step: u32) -> bool {
    let period = hyperball_polar_period() as u32;
    let polar_steps = hyperball_polar_numerator() as u32;
    step.saturating_sub(1) % period < polar_steps
}

pub(super) fn matrix_learning_rate(step: u32) -> f32 {
    if hyperball_enabled() {
        if hyperball_uses_polar(step) {
            hyperball_learning_rate()
        } else {
            muon_sign_learning_rate(step)
        }
    } else {
        muon_learning_rate(step)
    }
}

pub(super) fn muon_learning_rate(step: u32) -> f32 {
    let update_ratio = if sign_muon_uses_polar(step) {
        1.0
    } else {
        SIGN_MUON_SIGN_LR_RATIO
    };
    muon_learning_rate_with_ratio(step, update_ratio)
}

pub(super) fn muon_sign_learning_rate(step: u32) -> f32 {
    muon_learning_rate_with_ratio(step, SIGN_MUON_SIGN_LR_RATIO)
}

fn muon_learning_rate_with_ratio(step: u32, update_ratio: f32) -> f32 {
    MUON_LR
        * SIGN_MUON_FULL_LR_MULTIPLIER
        * update_ratio
        * super::learning_rate::muon_multiplier(step)
}

fn sign_muon_uses_polar(step: u32) -> bool {
    step.saturating_sub(1).is_multiple_of(SIGN_MUON_PERIOD)
}

pub(in crate::training) const fn max_matrix_len() -> usize {
    max3(
        GPT2_N_EMBD * GPT2_QKV,
        GPT2_MLP * GPT2_N_EMBD,
        NEXTLAT_INPUT * NEXTLAT_HIDDEN,
    )
}

pub(in crate::training) const fn max_matrix_dim() -> usize {
    max2(GPT2_N_EMBD, NEXTLAT_HIDDEN)
}

pub(in crate::training) const fn max_polar_cols() -> usize {
    max3(max2(GPT2_QKV, GPT2_MLP), NEXTLAT_INPUT, NEXTLAT_HIDDEN)
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn max3(a: usize, b: usize, c: usize) -> usize {
    max2(max2(a, b), c)
}

#[cfg(test)]
mod tests {
    use super::{
        hyperball_uses_polar, matrix_learning_rate, muon_vs_enabled, sign_muon_uses_polar,
    };

    #[test]
    fn sign_muon_period_starts_with_polar_and_alternates() {
        assert!(sign_muon_uses_polar(1));
        assert!(!sign_muon_uses_polar(2));
        assert!(sign_muon_uses_polar(3));
        assert!(!sign_muon_uses_polar(4));
    }

    #[test]
    fn matrix_learning_rate_defaults_to_hyperball_schedule() {
        if std::env::var_os("TRAIN_HYPERBALL").is_none()
            && std::env::var_os("TRAIN_HYPERBALL_LR").is_none()
            && std::env::var_os("TRAIN_HYPERBALL_POLAR_PERIOD").is_none()
            && std::env::var_os("TRAIN_HYPERBALL_POLAR_NUMERATOR").is_none()
        {
            assert_eq!(matrix_learning_rate(1), super::HYPERBALL_LR);
            assert_eq!(matrix_learning_rate(3), super::muon_sign_learning_rate(3));
        }
    }

    #[test]
    fn hyperball_defaults_to_two_polar_steps_out_of_three() {
        if std::env::var_os("TRAIN_HYPERBALL_POLAR_PERIOD").is_none()
            && std::env::var_os("TRAIN_HYPERBALL_POLAR_NUMERATOR").is_none()
        {
            assert!(hyperball_uses_polar(1));
            assert!(hyperball_uses_polar(2));
            assert!(!hyperball_uses_polar(3));
            assert!(hyperball_uses_polar(4));
        }
    }

    #[test]
    fn variance_adaptive_muon_is_default_candidate() {
        if std::env::var_os("TRAIN_MUON_VS").is_none() {
            assert!(muon_vs_enabled());
        }
    }
}
