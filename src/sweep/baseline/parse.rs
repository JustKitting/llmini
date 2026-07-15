use std::path::PathBuf;

use crate::env_file::{parsed, value};
use crate::sweep::candidate::Candidate;

use super::record::Record;

pub(super) fn record(text: &str) -> Option<Record> {
    Some(Record {
        val_loss: parsed(text, "VAL_LOSS")?,
        completed_steps: parsed(text, "COMPLETED_STEPS"),
        elapsed_s: parsed(text, "TRAIN_ELAPSED_S"),
        screen_loss: parsed(text, "SCREEN_LOSS"),
        screen_completed_steps: parsed(text, "SCREEN_COMPLETED_STEPS"),
        screen_elapsed_s: parsed(text, "SCREEN_ELAPSED_S"),
        screen_reason: value(text, "SCREEN_REASON").map(ToString::to_string),
        log_path: PathBuf::from(value(text, "LOG_PATH").unwrap_or("")),
        candidate: Candidate {
            batch_size: parsed(text, "GPT2_BATCH_SIZE")?,
            n_layer: parsed(text, "GPT2_N_LAYER")?,
            n_embd: parsed(text, "GPT2_N_EMBD")?,
            n_head: parsed(text, "GPT2_N_HEAD")?,
            muon_phases: parsed(text, "MUON_MATRIX_PHASES")?,
            muon_blocks: parsed(text, "MUON_COOPERATIVE_BLOCKS")?,
            lr_scale: parsed(text, "TRAIN_LR_SCALE")?,
            adam_lr_scale: parsed(text, "TRAIN_ADAM_LR_SCALE")?,
            nextlat_lr_scale: parsed(text, "TRAIN_NEXTLAT_LR_SCALE").unwrap_or(1.0),
            warmup_steps: parsed(text, "TRAIN_LR_WARMUP_STEPS")?,
            start_ratio: parsed(text, "TRAIN_LR_START_RATIO")?,
            amuse_beta1: parsed(text, "TRAIN_AMUSE_BETA1")?,
            amuse_rho: parsed(text, "TRAIN_AMUSE_RHO")?,
        },
    })
}
