use std::path::{Path, PathBuf};

use crate::AppResult;

use super::{
    TokenDataLoader, fineweb, manifest, shakespeare,
    source::{
        DATASET_FINEWEB, DATASET_SHAKESPEARE, DATASET_SYNTH, repeat_first_window,
        token_count_paths, training_dataset,
    },
    synth, tokens,
    validation::train_end,
};

impl TokenDataLoader {
    pub fn from_training_dataset() -> AppResult<(String, Self)> {
        Self::from_dataset(&training_dataset())
    }

    /// Read the existing native validation prefix without preparing training data.
    pub(in crate::training) fn fineweb_validation_tokens() -> AppResult<Vec<u16>> {
        let path = fineweb::first_val_shard()?;
        let tokens = tokens::read_u16_tokens(&path)?;
        super::validation::validation_windows(&path, &tokens, 0)
    }

    pub fn from_dataset(dataset: &str) -> AppResult<(String, Self)> {
        let loader = match dataset {
            DATASET_FINEWEB => Self::from_fineweb(),
            DATASET_SYNTH => Self::from_synth(),
            DATASET_SHAKESPEARE => Self::from_shakespeare(),
            dataset => Err(format!(
                "unknown training dataset {dataset}; expected fineweb, synth, or shakespeare"
            )
            .into()),
        }?;
        Ok((dataset.to_owned(), loader))
    }

    pub fn from_fineweb() -> AppResult<Self> {
        fineweb::ensure_shards()?;
        Self::with_fineweb_holdout(fineweb::train_shards()?)
    }

    pub(in crate::training) fn from_training_manifest(path: &Path) -> AppResult<(Self, String)> {
        let (shards, sha256) = manifest::read(path)?;
        Ok((Self::with_fineweb_holdout(shards)?, sha256))
    }

    fn with_fineweb_holdout(train_paths: Vec<PathBuf>) -> AppResult<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::data::validation::VALIDATION_WINDOWS;
    use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
    use std::{fs, time::SystemTime};

    #[test]
    fn manifest_shards_load_in_order_and_retry_a_failed_read() -> AppResult<()> {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_nanos();
        let fixture =
            std::env::temp_dir().join(format!("training-manifest-{}-{nonce}", std::process::id()));
        fs::create_dir(&fixture)?;
        let result = (|| -> AppResult<()> {
            let first = fixture.join("z.bin");
            let second = fixture.join("a.bin");
            let manifest_path = fixture.join("train.json");
            let batch_span = GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1;
            tokens::write_u16_tokens(&first, &vec![23; batch_span])?;
            tokens::write_u16_tokens(&second, &vec![29; batch_span])?;
            fs::write(
                &manifest_path,
                serde_json::to_vec(&serde_json::json!({
                    "tokenizer": llama2_tokenizer::TOKENIZER_NAME,
                    "shards": ["z.bin", "a.bin"],
                }))?,
            )?;

            let (shards, _) = manifest::read(&manifest_path)?;
            let validation = vec![31; VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1)];
            let mut loader = TokenDataLoader::from_train_paths(
                shards,
                Some((fixture.join("validation.bin"), validation.clone())),
                false,
                false,
            )?;
            loader.repeat_first_window = false;
            let batch = loader.next_batch()?;
            assert_eq!(batch.source, first);
            assert_eq!(batch.tokens[0], 23);

            tokens::write_u16_tokens(&second, &vec![u16::MAX; batch_span])?;
            assert!(
                loader
                    .next_batch()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("outside the model vocabulary")
            );
            tokens::write_u16_tokens(&second, &vec![29; batch_span])?;
            let batch = loader.next_batch()?;
            assert_eq!(batch.source, second);
            assert_eq!(batch.offset, 0);
            assert_eq!(batch.tokens[0], 29);
            assert_eq!(loader.validation_tokens()?, validation);
            Ok(())
        })();
        let cleanup = fs::remove_dir_all(&fixture);
        result?;
        cleanup?;
        Ok(())
    }
}
