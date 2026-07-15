use std::path::PathBuf;

mod factory;
mod fineweb;
mod loader;
mod shakespeare;
mod source;
mod synth;
mod tokens;
mod validation;

pub(in crate::training) use validation::VALIDATION_WINDOWS;

pub struct TokenDataLoader {
    path: PathBuf,
    train_paths: Vec<PathBuf>,
    train_path_index: usize,
    tokens: Vec<u16>,
    validation_tokens: Option<Vec<u16>>,
    validation_path: Option<PathBuf>,
    offset: usize,
    train_end: usize,
    total_train_tokens: usize,
    repeat_first_window: bool,
    wrap_train: bool,
}

pub struct TokenWindowBatch {
    pub tokens: Vec<u16>,
    pub source: PathBuf,
    pub offset: usize,
    pub batch_size: usize,
    pub seq_len: usize,
}
