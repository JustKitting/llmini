use std::fs;
use std::path::{Path, PathBuf};

use synth_prep::fineweb::{
    DATA_DIR, DEFAULT_TRAIN_SHARD_COUNT, SHARD_FILE_PREFIX, SHARD_SIZE, SHARDS_DIR,
    TOKENIZATION_MARKER,
};

use crate::AppResult;

pub(super) fn train_shards() -> AppResult<Vec<PathBuf>> {
    let shards = shards_for_split("train")?
        .into_iter()
        .filter(|path| is_full_shard(path))
        .collect::<Vec<_>>();
    if shards.is_empty() {
        return Err(format!(
            "no full FineWeb {} train shards found",
            llama2_tokenizer::TOKENIZER_NAME
        )
        .into());
    }
    Ok(shards)
}

pub(super) fn first_val_shard() -> AppResult<PathBuf> {
    shards_for_split("val")?.into_iter().next().ok_or_else(|| {
        format!(
            "no FineWeb {} val shards found in {}",
            llama2_tokenizer::TOKENIZER_NAME,
            shard_dir().display()
        )
        .into()
    })
}

pub(super) fn ensure_shards() -> AppResult<()> {
    if train_shards().is_ok_and(|shards| shards.len() >= DEFAULT_TRAIN_SHARD_COUNT)
        && first_val_shard().is_ok_and(|path| is_full_shard(&path))
        && shard_dir().join(TOKENIZATION_MARKER).exists()
    {
        return Ok(());
    }

    clear_shards()?;
    synth_prep::fineweb::parse_data_for_train_shards(DEFAULT_TRAIN_SHARD_COUNT)?;
    let train_shard_count = train_shards()?.len();
    if train_shard_count < DEFAULT_TRAIN_SHARD_COUNT {
        return Err(format!(
            "expected {DEFAULT_TRAIN_SHARD_COUNT} full FineWeb train shards, found {train_shard_count}"
        )
        .into());
    }
    first_val_shard().map(|_| ())
}

fn shards_for_split(split: &str) -> AppResult<Vec<PathBuf>> {
    let dir = shard_dir();
    let prefix = format!("{SHARD_FILE_PREFIX}_{split}_");
    if !dir.exists() {
        return Err(format!("{} does not exist", dir.display()).into());
    }

    let mut paths = fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".bin"))
    });
    paths.sort();
    Ok(paths)
}

fn shard_dir() -> PathBuf {
    Path::new(DATA_DIR).join(SHARDS_DIR)
}

fn is_full_shard(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.len() == (SHARD_SIZE * 2) as u64)
}

fn clear_shards() -> AppResult<()> {
    let dir = shard_dir();
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let remove = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                (name.starts_with(SHARD_FILE_PREFIX) && name.ends_with(".bin"))
                    || name == TOKENIZATION_MARKER
            });
        if remove {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}
