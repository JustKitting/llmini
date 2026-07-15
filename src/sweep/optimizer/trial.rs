use std::collections::HashSet;

use crate::sweep::{candidate::Candidate, config::SweepConfig, history::Trial};

const NAN_PENALTY_LOSS: f64 = 1.0e6;
const FAILED_TRIAL_PENALTY_LOSS: f64 = 5.0e5;

pub(super) fn best_local_center(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    best_screen_candidate(trials, config).or_else(|| best_full_candidate(trials, config))
}

fn best_screen_candidate(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    trials
        .iter()
        .filter_map(|trial| {
            let loss = trial.screen_val_loss?;
            if !loss.is_finite() || !trial.candidate.meets_model_floor() {
                return None;
            }
            if !time_budget_matches(trial.screen_elapsed_s, config.screen_max_seconds) {
                return None;
            }
            Some((loss, trial.candidate.with_model_floor()))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, candidate)| candidate)
}

fn best_full_candidate(trials: &[Trial], config: &SweepConfig) -> Option<Candidate> {
    trials
        .iter()
        .filter_map(|trial| {
            let loss = trial.val_loss?;
            if !loss.is_finite() || !trial.candidate.meets_model_floor() {
                return None;
            }
            if !time_budget_matches(trial.elapsed_s, config.max_seconds) {
                return None;
            }
            Some((loss, trial.candidate.with_model_floor()))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, candidate)| candidate)
}

fn time_budget_matches(elapsed_s: Option<f64>, target_s: f64) -> bool {
    let Some(elapsed_s) = elapsed_s else {
        return false;
    };
    elapsed_s.is_finite() && elapsed_s >= target_s * 0.8 && elapsed_s <= target_s * 1.25
}

pub(super) fn observed_loss(trial: &Trial) -> Option<f64> {
    if !trial.candidate.meets_model_floor() {
        return None;
    }
    if trial.status == "dry_run" {
        return None;
    }
    if trial.status == "failed_build" || trial.status == "failed_run" {
        return Some(FAILED_TRIAL_PENALTY_LOSS);
    }
    if trial.status == "rejected_screen" {
        return trial.screen_val_loss.or(Some(FAILED_TRIAL_PENALTY_LOSS));
    }
    if trial.status.starts_with("nan") {
        return Some(NAN_PENALTY_LOSS);
    }
    trial.val_loss
}

pub(super) fn infeasible_build_shapes(trials: &[Trial], config: &SweepConfig) -> HashSet<String> {
    trials
        .iter()
        .filter(|trial| trial.status == "failed_build" || trial.status == "failed_run")
        .filter(|trial| {
            let elapsed = trial.screen_elapsed_s.or(trial.elapsed_s).unwrap_or(0.0);
            elapsed == 0.0 || elapsed >= config.screen_max_seconds * 0.95
        })
        .map(|trial| trial.candidate.build_key())
        .collect()
}
