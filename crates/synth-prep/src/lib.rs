use std::error::Error;

pub mod fineweb;
mod huggingface;
mod parquet_text;
mod pipeline;
mod shards;
pub mod synth;
mod tokenize;

pub type AppResult<T> = Result<T, Box<dyn Error>>;

pub use pipeline::{parse_data, parse_data_for_train_shards};
pub use synth::{
    DATA_DIR, DATASET_NAME, DATASET_OWNER, DATASET_REPO, DATASET_SPLIT, DEFAULT_TRAIN_SHARD_COUNT,
    PARQUET_FILE_PATTERN, SHARD_FILE_PREFIX, SHARD_SIZE, TOKENIZATION_MARKER,
};
