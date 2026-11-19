//! Read the same token windows as the native training loader.
use super::super::data::TrainingWindowLayout;
use crate::AppResult;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) struct InputWindow {
    pub tokens: Vec<u16>,
    pub metadata: Value,
}

pub(super) fn load(
    shard: &Path,
    offset: usize,
    batch_size: usize,
    seq_len: usize,
) -> AppResult<InputWindow> {
    let layout = TrainingWindowLayout::for_shard(shard, batch_size, seq_len)?;
    let (tokens, bytes) = layout.read(shard, offset)?;
    Ok(InputWindow {
        tokens,
        metadata: json!({"source":shard,"token_offset":offset,"batch_size":batch_size,
            "seq_len":seq_len,"source_tokens":layout.source_tokens(),
            "source_window_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "layout":layout.description()}),
    })
}
