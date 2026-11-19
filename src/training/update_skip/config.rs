use super::super::env::{env_bool, env_f32, env_usize};

const ENABLED_ENV: &str = "TRAIN_SKIP_UNSTABLE_UPDATES";
const ROLLING_INTERVAL_ENV: &str = "TRAIN_SKIP_ROLLING_INTERVAL";
const SIGMA_FACTOR_ENV: &str = "TRAIN_SKIP_SIGMA_FACTOR";
const USE_LOSS_ENV: &str = "TRAIN_SKIP_USE_LOSS";
const USE_GRAD_NORM_ENV: &str = "TRAIN_SKIP_USE_GRAD_NORM";
const CAP_GRAD_HISTORY_ENV: &str = "TRAIN_SKIP_CAP_GRAD_HISTORY";

const DEFAULT_ENABLED: bool = true;
const DEFAULT_ROLLING_INTERVAL: usize = 128;
const DEFAULT_SIGMA_FACTOR: f32 = 6.0;

#[derive(Clone, Copy)]
pub(super) struct UpdateSkipConfig {
    pub(super) enabled: bool,
    pub(super) rolling_interval: usize,
    pub(super) sigma_factor: f32,
    pub(super) use_loss: bool,
    pub(super) use_grad_norm: bool,
    pub(super) cap_grad_history: bool,
}

impl UpdateSkipConfig {
    pub(super) fn from_env() -> Self {
        Self {
            enabled: env_bool(ENABLED_ENV).unwrap_or(DEFAULT_ENABLED),
            rolling_interval: env_usize(ROLLING_INTERVAL_ENV)
                .unwrap_or(DEFAULT_ROLLING_INTERVAL)
                .max(2),
            sigma_factor: env_f32(SIGMA_FACTOR_ENV)
                .filter(|value| value.is_finite())
                .unwrap_or(DEFAULT_SIGMA_FACTOR)
                .max(0.0),
            use_loss: env_bool(USE_LOSS_ENV).unwrap_or(true),
            use_grad_norm: env_bool(USE_GRAD_NORM_ENV).unwrap_or(true),
            cap_grad_history: env_bool(CAP_GRAD_HISTORY_ENV).unwrap_or(false),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test() -> Self {
        Self {
            enabled: true,
            rolling_interval: 4,
            sigma_factor: 2.0,
            use_loss: true,
            use_grad_norm: true,
            cap_grad_history: false,
        }
    }
}
