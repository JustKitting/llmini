use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};

use super::{
    TokenDataLoader, TokenWindowBatch, TrainingWindowLayout, tokens,
    validation::{VALIDATION_WINDOWS, train_end, validation_windows},
};
use crate::AppResult;

impl TokenDataLoader {
    pub fn next_batch(&mut self) -> AppResult<TokenWindowBatch> {
        let len = self.layout.row_len();
        if self.train_end < len {
            return Err(format!("{} has fewer than {len} tokens", self.path.display()).into());
        }
        if !self.repeat_first_window && self.offset + self.layout.source_tokens() > self.train_end {
            self.advance_train_shard()?;
        }

        let batch_offset = self.offset;
        let stored = &self.tokens[batch_offset..];
        let tokens = if self.repeat_first_window {
            stored[..len].repeat(GPT2_BATCH_SIZE)
        } else {
            self.offset += self.layout.next_batch_stride();
            self.layout.flatten(stored)
        };
        Ok(TokenWindowBatch {
            tokens,
            source: self.path.clone(),
            offset: batch_offset,
            batch_size: GPT2_BATCH_SIZE,
            seq_len: GPT2_SEQ_LEN,
        })
    }

    pub fn token_count(&self) -> usize {
        self.total_train_tokens
    }

    pub fn validation_tokens(&self) -> AppResult<Vec<u16>> {
        if let Some(tokens) = &self.validation_tokens {
            let path = self.validation_path.as_deref().unwrap_or(&self.path);
            return validation_windows(path, tokens, 0);
        }

        let needed = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
        let start = self.tokens.len().saturating_sub(needed);
        validation_windows(&self.path, &self.tokens, start)
    }

    fn advance_train_shard(&mut self) -> AppResult<()> {
        let next_index = self.train_path_index + 1;
        let next_index = if next_index >= self.train_paths.len() {
            if !self.wrap_train {
                return Err(format!(
                    "ran out of fresh train shards after {}; prepare more train shards",
                    self.path.display()
                )
                .into());
            }
            0
        } else {
            next_index
        };

        let path = &self.train_paths[next_index];
        let tokens = tokens::read_u16_tokens(path)?;
        let layout = TrainingWindowLayout::for_shard(path, GPT2_BATCH_SIZE, GPT2_SEQ_LEN)?;
        let train_end = if self.validation_tokens.is_some() {
            tokens.len()
        } else {
            train_end(tokens.len())
        };
        let batch_span = layout.source_tokens();
        if train_end < batch_span {
            return Err(format!(
                "{} has fewer than {batch_span} training tokens",
                path.display()
            )
            .into());
        }

        self.train_path_index = next_index;
        self.path = path.clone();
        self.tokens = tokens;
        self.train_end = train_end;
        self.layout = layout;
        self.offset = 0;
        Ok(())
    }
}
