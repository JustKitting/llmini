use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use crate::AppResult;

/// A stream overlaps adjacent rows by one token. A windows shard stores the
/// last target of each row explicitly so rows can be regrouped into batches.
#[derive(Clone, Copy)]
pub(in crate::training) struct TrainingWindowLayout {
    batch_size: usize,
    row_len: usize,
    row_stride: usize,
    source_tokens: usize,
}

impl TrainingWindowLayout {
    pub(in crate::training) fn for_shard(
        path: &Path,
        batch_size: usize,
        seq_len: usize,
    ) -> AppResult<Self> {
        if batch_size == 0 || seq_len == 0 {
            return Err("training window requires nonzero batch and sequence lengths".into());
        }
        let row_len = seq_len
            .checked_add(1)
            .ok_or("training row length overflow")?;
        batch_size
            .checked_mul(row_len)
            .ok_or("training batch length overflow")?;
        let row_stride = if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".windows.bin"))
        {
            row_len
        } else {
            seq_len
        };
        let source_tokens = (batch_size - 1) * row_stride + row_len;
        Ok(Self {
            batch_size,
            row_len,
            row_stride,
            source_tokens,
        })
    }

    fn independent(self) -> bool {
        self.row_stride == self.row_len
    }

    pub(super) fn row_len(self) -> usize {
        self.row_len
    }

    pub(super) fn next_batch_stride(self) -> usize {
        self.batch_size * self.row_stride
    }

    pub(in crate::training) fn source_tokens(self) -> usize {
        self.source_tokens
    }

    pub(in crate::training) fn description(self) -> &'static str {
        if self.independent() {
            "batch_size independent rows, each storing seq_len inputs and its last target"
        } else {
            "batch_size adjacent seq_len-token sequences, each with its following target token"
        }
    }

    pub(super) fn validate_shard(self, path: &Path, byte_count: u64) -> AppResult {
        if !byte_count.is_multiple_of(2) || byte_count / 2 < self.source_tokens as u64 {
            return Err(format!(
                "{} must contain at least one full native u16 training batch",
                path.display()
            )
            .into());
        }
        if self.independent() && !(byte_count / 2).is_multiple_of(self.source_tokens as u64) {
            return Err(format!(
                "{} must contain whole batches of independent training windows",
                path.display()
            )
            .into());
        }
        Ok(())
    }

    pub(super) fn training_input_tokens(self, stored_tokens: usize) -> usize {
        if self.independent() {
            stored_tokens / self.row_len * (self.row_len - 1)
        } else {
            stored_tokens
        }
    }

    pub(super) fn flatten(self, stored: &[u16]) -> Vec<u16> {
        if self.independent() {
            return stored[..self.source_tokens].to_vec();
        }
        let mut tokens = Vec::with_capacity(self.batch_size * self.row_len);
        for row in 0..self.batch_size {
            let start = row * self.row_stride;
            tokens.extend_from_slice(&stored[start..start + self.row_len]);
        }
        tokens
    }

    pub(in crate::training) fn read(
        self,
        shard: &Path,
        offset: usize,
    ) -> AppResult<(Vec<u16>, Vec<u8>)> {
        if self.independent() && !offset.is_multiple_of(self.row_stride) {
            return Err("token offset must start on an independent row boundary".into());
        }
        let byte_offset = offset.checked_mul(2).ok_or("token offset overflow")?;
        let byte_len = self
            .source_tokens
            .checked_mul(2)
            .ok_or("training window byte length overflow")?;
        let byte_end = byte_offset
            .checked_add(byte_len)
            .ok_or("training window byte range overflow")?;
        let mut file = File::open(shard)?;
        let file_len = file.metadata()?.len();
        if !file_len.is_multiple_of(2) || u64::try_from(byte_end)? > file_len {
            return Err(format!(
                "training window at token {offset} requires {} valid u16 tokens in {}",
                self.source_tokens,
                shard.display()
            )
            .into());
        }
        file.seek(SeekFrom::Start(u64::try_from(byte_offset)?))?;
        let mut bytes = vec![0; byte_len];
        file.read_exact(&mut bytes)?;
        let stored = bytes
            .chunks_exact(2)
            .map(|v| u16::from_ne_bytes([v[0], v[1]]))
            .collect::<Vec<_>>();
        if stored
            .iter()
            .any(|&id| usize::from(id) >= llama2_tokenizer::MODEL_VOCAB_SIZE)
        {
            return Err("training window contains a token outside the model vocabulary".into());
        }
        Ok((self.flatten(&stored), bytes))
    }
}
