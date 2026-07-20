pub const DATASET_OWNER: &str = "PleIAs";
pub const DATASET_NAME: &str = "SYNTH";
pub const DATASET_REPO: &str = "PleIAs/SYNTH";
pub const DATASET_SPLIT: &str = "train";
pub const DATA_DIR: &str = "data/synth";
pub const PARQUET_DIR: &str = "parquet";
pub const SHARDS_DIR: &str = "shards";
pub const SHARD_SIZE: usize = 100_000_000;
pub const DEFAULT_TRAIN_SHARD_COUNT: usize = 4;
pub const SHARD_FILE_PREFIX: &str = SELECTED_SYNTH_SHARD_FILE_PREFIX;
pub const TOKENIZATION_MARKER: &str = SELECTED_SYNTH_TOKENIZATION_MARKER;
pub const PARQUET_FILE_PATTERN: &str = "synth_*.parquet";
use llama2_tokenizer::{
    SYNTH_SHARD_FILE_PREFIX as SELECTED_SYNTH_SHARD_FILE_PREFIX,
    SYNTH_TOKENIZATION_MARKER as SELECTED_SYNTH_TOKENIZATION_MARKER,
};
