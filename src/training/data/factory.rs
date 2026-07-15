use std::path::PathBuf;

use crate::AppResult;

use super::{
    TokenDataLoader, fineweb, shakespeare,
    source::{
        DATASET_FINEWEB, DATASET_SHAKESPEARE, DATASET_SYNTH, repeat_first_window,
        token_count_paths, training_dataset,
    },
    synth, tokens,
    validation::train_end,
};

impl TokenDataLoader {
    pub fn from_training_dataset() -> AppResult<(String, Self)> {
        let dataset = training_dataset();
        let loader = match dataset.as_str() {
            DATASET_FINEWEB => Self::from_fineweb(),
            DATASET_SYNTH => Self::from_synth(),
            DATASET_SHAKESPEARE => Self::from_shakespeare(),
            dataset => Err(format!(
                "unknown TRAIN_DATASET={dataset}; expected fineweb, synth, or shakespeare"
            )
            .into()),
        }?;
        Ok((dataset, loader))
    }

    pub fn from_fineweb() -> AppResult<Self> {
        fineweb::ensure_shards()?;
        let train_paths = fineweb::train_shards()?;
        let validation_path = fineweb::first_val_shard()?;
        let validation_tokens = tokens::read_u16_tokens(&validation_path)?;
        Self::from_train_paths(
            train_paths,
            Some((validation_path, validation_tokens)),
            false,
            false,
        )
    }

    pub fn from_synth() -> AppResult<Self> {
        synth::ensure_shards()?;
        let train_paths = synth::train_shards()?;
        let validation_path = synth::first_val_shard()?;
        let validation_tokens = tokens::read_u16_tokens(&validation_path)?;
        Self::from_train_paths(
            train_paths,
            Some((validation_path, validation_tokens)),
            false,
            false,
        )
    }

    pub fn from_shakespeare() -> AppResult<Self> {
        let path = shakespeare::ensure_shard()?;
        Self::from_train_paths(vec![path], None, true, true)
    }

    fn from_train_paths(
        train_paths: Vec<PathBuf>,
        validation: Option<(PathBuf, Vec<u16>)>,
        wrap_train: bool,
        reserve_validation_tail: bool,
    ) -> AppResult<Self> {
        let path = train_paths
            .first()
            .cloned()
            .ok_or("training dataset has no train shards")?;
        let tokens = tokens::read_u16_tokens(&path)?;
        let train_end = if validation.is_none() && reserve_validation_tail {
            train_end(tokens.len())
        } else {
            tokens.len()
        };
        let total_train_tokens = token_count_paths(&train_paths)?;
        let (validation_path, validation_tokens) = validation.unzip();
        Ok(Self {
            path,
            train_paths,
            train_path_index: 0,
            tokens,
            validation_tokens,
            validation_path,
            offset: 0,
            train_end,
            total_train_tokens,
            repeat_first_window: repeat_first_window(),
            wrap_train,
        })
    }
}
