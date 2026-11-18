use burn::serde::Deserialize;
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::AppResult;

#[derive(Deserialize)]
#[serde(crate = "burn::serde")]
struct Manifest {
    tokenizer: String,
    shards: Vec<PathBuf>,
}

pub(super) fn read(path: &Path) -> AppResult<(Vec<PathBuf>, String)> {
    let contents = fs::read(path)?;
    let manifest: Manifest = serde_json::from_slice(&contents)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if manifest.tokenizer != llama2_tokenizer::TOKENIZER_NAME {
        return Err(format!(
            "{} uses tokenizer {}, expected {}",
            path.display(),
            manifest.tokenizer,
            llama2_tokenizer::TOKENIZER_NAME
        )
        .into());
    }
    if manifest.shards.is_empty() {
        return Err(format!("{} has no training shards", path.display()).into());
    }
    let root = path.parent().unwrap_or(Path::new("."));
    let min_bytes = 2 * (GPT2_BATCH_SIZE * GPT2_SEQ_LEN + 1) as u64;
    let mut shards = Vec::with_capacity(manifest.shards.len());
    for name in manifest.shards {
        let shard = root.join(name);
        let info = shard
            .metadata()
            .map_err(|error| format!("{}: {error}", shard.display()))?;
        if !info.is_file() || info.len() < min_bytes || !info.len().is_multiple_of(2) {
            return Err(format!(
                "{} must contain at least one full native u16 training batch",
                shard.display()
            )
            .into());
        }
        shards.push(shard);
    }
    Ok((shards, format!("{:x}", Sha256::digest(&contents))))
}
