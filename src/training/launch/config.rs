use std::path::PathBuf;

use super::super::SamplingConfig;
pub(in crate::training) use crate::training::env::{env_bool, env_nonempty};
use crate::training::env::{env_f32, env_f64, env_u64, env_usize};

const DEFAULT_SEED: u64 = 0x4750_5432;
const DEFAULT_TRAIN_MAX_SECONDS: f64 = 900.0;
const DEFAULT_TRAIN_STEP_CAP: usize = 1_000_000;
const AUTO_GENERATE_MIN_TRAIN_SECONDS: f64 = 900.0;
const DEFAULT_SYNTH_PROMPT: &str = "The";
const DEFAULT_SHAKESPEARE_PROMPT: &str = "KING:";

#[derive(Clone, Copy)]
pub(in crate::training) struct TrainConfig {
    pub(in crate::training) seed: u64,
    pub(in crate::training) latent_feedback: bool,
    pub(in crate::training) mhc: bool,
    pub(in crate::training) engram_mhc: bool,
    pub(in crate::training) engram_table_params: Option<usize>,
    pub(in crate::training) engram_four_grams: bool,
    pub(in crate::training) engram_second_layer: Option<usize>,
    pub(in crate::training) step_cap: usize,
    pub(in crate::training) log_interval: usize,
    pub(in crate::training) eval_interval: Option<usize>,
    pub(in crate::training) eval_repeats: usize,
    pub(in crate::training) max_seconds: f64,
}

impl TrainConfig {
    pub(super) fn engram_options(&self) -> crate::training::engram::MultistreamOptions {
        crate::training::engram::MultistreamOptions {
            requested_values: self.engram_table_params,
            four_grams: self.engram_four_grams,
            second_layer: self.engram_second_layer,
        }
    }

    pub(super) fn from_env() -> Self {
        Self {
            seed: env_u64("TRAIN_SEED").unwrap_or(DEFAULT_SEED),
            latent_feedback: env_bool("TRAIN_LATENT_FEEDBACK").unwrap_or(false),
            mhc: super::super::mhc::enabled(),
            engram_mhc: super::super::engram::multistream_enabled(),
            engram_table_params: None,
            engram_four_grams: false,
            engram_second_layer: None,
            step_cap: env_usize("TRAIN_STEPS").unwrap_or(DEFAULT_TRAIN_STEP_CAP),
            log_interval: env_usize("TRAIN_LOG_INTERVAL").unwrap_or(1).max(1),
            eval_interval: env_usize("TRAIN_EVAL_INTERVAL").filter(|interval| *interval > 0),
            eval_repeats: env_usize("TRAIN_EVAL_REPEATS").unwrap_or(1).max(1),
            max_seconds: env_f64("TRAIN_MAX_SECONDS")
                .filter(|seconds| *seconds > 0.0)
                .unwrap_or(DEFAULT_TRAIN_MAX_SECONDS),
        }
    }
}

pub(super) fn load_model_path() -> Option<PathBuf> {
    env_nonempty("TRAIN_LOAD_MODEL").map(PathBuf::from)
}

pub(super) fn generate_prompt(dataset: &str, train_elapsed_s: f64) -> Option<String> {
    if env_bool("TRAIN_GENERATE").is_some_and(|enabled| !enabled) {
        return None;
    }
    env_nonempty("TRAIN_GENERATE_PROMPT").or_else(|| {
        (train_elapsed_s >= AUTO_GENERATE_MIN_TRAIN_SECONDS)
            .then(|| default_generate_prompt(dataset).to_string())
    })
}

pub(super) fn generate_tokens() -> usize {
    env_usize("TRAIN_GENERATE_TOKENS").unwrap_or(128)
}

pub(super) fn sampling_config() -> SamplingConfig {
    SamplingConfig {
        temperature: env_f32("TRAIN_GENERATE_TEMPERATURE").unwrap_or(0.7),
        top_k: env_usize("TRAIN_GENERATE_TOP_K").unwrap_or(32),
        top_p: env_f32("TRAIN_GENERATE_TOP_P").unwrap_or(0.9),
    }
}

pub(super) fn should_log_step(step: usize, step_cap: usize, log_interval: usize) -> bool {
    step == 0 || step + 1 == step_cap || step.is_multiple_of(log_interval)
}

pub(super) fn should_eval_step(step: usize, step_cap: usize, eval_interval: Option<usize>) -> bool {
    let completed_steps = step + 1;
    eval_interval.is_some_and(|interval| {
        completed_steps < step_cap && completed_steps.is_multiple_of(interval)
    })
}

fn default_generate_prompt(dataset: &str) -> &'static str {
    match dataset {
        "shakespeare" => DEFAULT_SHAKESPEARE_PROMPT,
        _ => DEFAULT_SYNTH_PROMPT,
    }
}

#[cfg(test)]
mod tests {
    use super::should_eval_step;

    #[test]
    fn periodic_validation_uses_completed_step_numbers() {
        assert!(!should_eval_step(0, 200, Some(50)));
        assert!(should_eval_step(49, 200, Some(50)));
        assert!(should_eval_step(99, 200, Some(50)));
        assert!(should_eval_step(149, 200, Some(50)));
    }

    #[test]
    fn final_periodic_validation_is_left_to_final_eval() {
        assert!(!should_eval_step(199, 200, Some(50)));
        assert!(!should_eval_step(99, 100, Some(10)));
    }
}
