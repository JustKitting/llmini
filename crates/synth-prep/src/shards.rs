use std::path::PathBuf;

use bytemuck::cast_slice;

use crate::AppResult;
use crate::synth::{DATA_DIR, SHARD_FILE_PREFIX, SHARD_SIZE, SHARDS_DIR};

pub struct ShardWriter {
    output_dir: PathBuf,
    file_prefix: String,
    shard_size: usize,
    shard_index: usize,
    target_train_shards: usize,
    tokens: Vec<u16>,
}

impl ShardWriter {
    pub fn new(target_train_shards: usize) -> Self {
        Self::for_dataset(
            PathBuf::from(DATA_DIR).join(SHARDS_DIR),
            SHARD_FILE_PREFIX,
            SHARD_SIZE,
            target_train_shards,
        )
    }

    pub fn for_dataset(
        output_dir: PathBuf,
        file_prefix: impl Into<String>,
        shard_size: usize,
        target_train_shards: usize,
    ) -> Self {
        Self {
            output_dir,
            file_prefix: file_prefix.into(),
            shard_size,
            shard_index: 0,
            target_train_shards,
            tokens: Vec::with_capacity(shard_size),
        }
    }

    pub fn push(&mut self, token: u16) -> AppResult<()> {
        self.tokens.push(token);
        if self.tokens.len() == self.shard_size {
            self.flush_current()?;
        }
        Ok(())
    }

    pub fn has_required_train_and_val_shards(&self) -> bool {
        self.shard_index > self.target_train_shards
    }

    pub fn finish(mut self) -> AppResult<()> {
        if !self.has_required_train_and_val_shards() && !self.tokens.is_empty() {
            self.flush_current()?;
        }
        Ok(())
    }

    fn flush_current(&mut self) -> AppResult<()> {
        let split = if self.shard_index == 0 {
            "val"
        } else {
            "train"
        };
        let path = self.output_dir.join(format!(
            "{}_{split}_{:06}.bin",
            self.file_prefix, self.shard_index
        ));

        std::fs::write(path, cast_slice(&self.tokens))?;
        self.tokens.clear();
        self.shard_index += 1;
        Ok(())
    }
}
