use std::path::PathBuf;

use super::TrainingWindowLayout;
use crate::AppResult;
use crate::training::env::{env_bool, env_nonempty};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

const TRAIN_DATASET_ENV: &str = "TRAIN_DATASET";
const TRAIN_REPEAT_BATCH_ENV: &str = "TRAIN_REPEAT_BATCH";

pub(super) const DATASET_SYNTH: &str = "synth";
pub(super) const DATASET_FINEWEB: &str = "fineweb";
pub(super) const DATASET_SHAKESPEARE: &str = "shakespeare";

pub(super) fn training_dataset() -> String {
    env_nonempty(TRAIN_DATASET_ENV).unwrap_or_else(|| DATASET_FINEWEB.to_string())
}

pub(super) fn repeat_first_window() -> bool {
    env_bool(TRAIN_REPEAT_BATCH_ENV).unwrap_or(false)
}

pub(super) fn token_count_paths(paths: &[PathBuf]) -> AppResult<usize> {
    let mut total = 0usize;
    for path in paths {
        let bytes = path.metadata()?.len();
        if bytes % 2 != 0 {
            return Err(format!("{} has odd byte length", path.display()).into());
        }
        let layout = TrainingWindowLayout::for_shard(path, GPT2_BATCH_SIZE, GPT2_SEQ_LEN)?;
        total += layout.training_input_tokens(usize::try_from(bytes / 2)?);
    }
    Ok(total)
}
