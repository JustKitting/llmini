use std::fs;
use std::path::{Path, PathBuf};

use synth_prep::synth::SHARDS_DIR;
use synth_prep::{
    DATA_DIR, DEFAULT_TRAIN_SHARD_COUNT, SHARD_FILE_PREFIX, SHARD_SIZE, TOKENIZATION_MARKER,
};

use crate::AppResult;

pub(super) fn train_shards() -> AppResult<Vec<PathBuf>> {
    let shards = shards_for_split("train")?
        .into_iter()
        .filter(|path| is_full_synth_shard(path))
        .collect::<Vec<_>>();
    if shards.is_empty() {
        return Err("no full SYNTH train shards found".into());
    }
    Ok(shards)
}

pub(super) fn first_val_shard() -> AppResult<PathBuf> {
    shards_for_split("val")?
        .into_iter()
        .next()
        .ok_or_else(|| format!("no val shards found in {}", synth_shard_dir().display()).into())
}

pub(super) fn ensure_shards() -> AppResult<()> {
    if train_shards().is_ok_and(|shards| shards.len() >= DEFAULT_TRAIN_SHARD_COUNT)
        && first_val_shard().is_ok()
        && synth_tokenization_marker().exists()
    {
        return Ok(());
    }

    clear_shards()?;
    synth_prep::parse_data_for_train_shards(DEFAULT_TRAIN_SHARD_COUNT)?;
    let train_shard_count = train_shards()?.len();
    if train_shard_count < DEFAULT_TRAIN_SHARD_COUNT {
        return Err(format!(
            "expected {DEFAULT_TRAIN_SHARD_COUNT} full SYNTH train shards, found {train_shard_count}"
        )
        .into());
    }
    first_val_shard().map(|_| ())
}

fn shards_for_split(split: &str) -> AppResult<Vec<PathBuf>> {
    let dir = synth_shard_dir();
    let prefix = format!("{SHARD_FILE_PREFIX}_{split}_");
    if !dir.exists() {
        return Err(format!("{} does not exist after SYNTH prep", dir.display()).into());
    }

    let mut shards = matching_entries(&dir, |file_name| is_bin_shard(file_name, &prefix))?;
    shards.sort();
    Ok(shards)
}

fn synth_shard_dir() -> PathBuf {
    Path::new(DATA_DIR).join(SHARDS_DIR)
}

fn is_full_synth_shard(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.len() == (SHARD_SIZE * 2) as u64)
}

fn synth_tokenization_marker() -> PathBuf {
    synth_shard_dir().join(TOKENIZATION_MARKER)
}

fn clear_shards() -> AppResult<()> {
    let dir = synth_shard_dir();
    if !dir.exists() {
        return Ok(());
    }

    for path in matching_entries(&dir, |file_name| {
        is_bin_shard(file_name, SHARD_FILE_PREFIX) || file_name == TOKENIZATION_MARKER
    })? {
        fs::remove_file(path)?;
    }

    Ok(())
}

fn is_bin_shard(file_name: &str, prefix: &str) -> bool {
    file_name.starts_with(prefix) && file_name.ends_with(".bin")
}

fn matching_entries(dir: &Path, keep: impl Fn(&str) -> bool) -> AppResult<Vec<PathBuf>> {
    let mut paths = fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(&keep)
    });
    Ok(paths)
}
