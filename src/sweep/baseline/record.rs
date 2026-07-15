use std::path::PathBuf;

use crate::sweep::{candidate::Candidate, history::Trial};

#[derive(Clone)]
pub(super) struct Record {
    pub(super) val_loss: f64,
    pub(super) completed_steps: Option<usize>,
    pub(super) elapsed_s: Option<f64>,
    pub(super) screen_loss: Option<f64>,
    pub(super) screen_completed_steps: Option<usize>,
    pub(super) screen_elapsed_s: Option<f64>,
    pub(super) screen_reason: Option<String>,
    pub(super) log_path: PathBuf,
    pub(super) candidate: Candidate,
}

impl Record {
    pub(super) fn measured_trial(&self) -> Trial {
        Trial {
            candidate: self.candidate.clone(),
            status: "success".to_string(),
            val_loss: Some(self.val_loss),
            completed_steps: self.completed_steps,
            elapsed_s: self.elapsed_s,
            screen_val_loss: self.screen_loss,
            screen_completed_steps: self.screen_completed_steps,
            screen_elapsed_s: self.screen_elapsed_s,
            screen_reason: self.screen_reason.clone(),
            log_path: self.log_path.clone(),
        }
    }

    pub(super) fn from_trial(trial: &Trial) -> Option<Self> {
        if trial.status != "success" || !trial.candidate.meets_model_floor() {
            return None;
        }
        Some(Self {
            val_loss: trial.val_loss.filter(|loss| loss.is_finite())?,
            completed_steps: trial.completed_steps,
            elapsed_s: trial.elapsed_s,
            screen_loss: trial.screen_val_loss,
            screen_completed_steps: trial.screen_completed_steps,
            screen_elapsed_s: trial.screen_elapsed_s,
            screen_reason: trial.screen_reason.clone(),
            log_path: trial.log_path.clone(),
            candidate: trial.candidate.clone(),
        })
    }
}
