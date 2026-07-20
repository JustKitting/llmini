use std::fs;

use llama2_tokenizer::{SYNTH_TOKENIZATION_MARKER, TrainingTokenizer};

use crate::synth::{DATA_DIR, DEFAULT_TRAIN_SHARD_COUNT, PARQUET_DIR, SHARDS_DIR};
use crate::{AppResult, huggingface, parquet_text, shards};

pub fn parse_data() -> AppResult<()> {
    parse_data_for_train_shards(DEFAULT_TRAIN_SHARD_COUNT)
}

pub fn parse_data_for_train_shards(target_train_shards: usize) -> AppResult<()> {
    let files = huggingface::list_parquet_files()?;
    if files.is_empty() {
        return Err("PleIAs/SYNTH did not list any synth_*.parquet files".into());
    }
    let data_dir = std::path::PathBuf::from(DATA_DIR);

    fs::create_dir_all(data_dir.join(PARQUET_DIR))?;
    fs::create_dir_all(data_dir.join(SHARDS_DIR))?;

    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let mut writer = shards::ShardWriter::new(target_train_shards);

    for file in &files {
        let path = huggingface::download_parquet_file(file)?;
        parquet_text::tokenize_parquet_file(&path, &tokenizer, &mut writer)?;
        if writer.has_required_train_and_val_shards() {
            break;
        }
    }

    writer.finish()?;
    std::fs::write(
        data_dir.join(SHARDS_DIR).join(SYNTH_TOKENIZATION_MARKER),
        b"1\n",
    )?;
    Ok(())
}
