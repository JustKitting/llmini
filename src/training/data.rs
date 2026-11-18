use std::path::PathBuf;

mod factory;
mod fineweb;
mod loader;
mod manifest;
mod shakespeare;
mod source;
mod synth;
mod tokens;
mod validation;
mod windows;

pub(in crate::training) use validation::{VALIDATION_WINDOWS, validation_target_byte_count};
pub(in crate::training) use windows::TrainingWindowLayout;

pub struct TokenDataLoader {
    path: PathBuf,
    train_paths: Vec<PathBuf>,
    train_path_index: usize,
    tokens: Vec<u16>,
    validation_tokens: Option<Vec<u16>>,
    validation_path: Option<PathBuf>,
    offset: usize,
    layout: TrainingWindowLayout,
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
