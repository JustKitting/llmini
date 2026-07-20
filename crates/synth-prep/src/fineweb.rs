use std::fs::{self, File};
use std::path::{Path, PathBuf};

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use hf_hub::HFClientSync;
use llama2_tokenizer::{
    DOCUMENT_BOUNDARIES, FINEWEB_SHARD_FILE_PREFIX, FINEWEB_TOKENIZATION_MARKER, MODEL_VOCAB_SIZE,
    TOKENIZER_NAME, TOKENIZER_VOCAB_SIZE, TrainingTokenizer,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::AppResult;
use crate::shards::ShardWriter;
use crate::tokenize::tokenize_doc;

pub const DATASET_OWNER: &str = "HuggingFaceFW";
pub const DATASET_NAME: &str = "fineweb";
pub const DATASET_REPO: &str = "HuggingFaceFW/fineweb";
pub const DATASET_CONFIG: &str = "sample-10BT";
pub const DATA_DIR: &str = "data/fineweb";
pub const PARQUET_DIR: &str = "parquet";
pub const SHARDS_DIR: &str = "shards";
pub const SHARD_SIZE: usize = 25_000_000;
pub const DEFAULT_TRAIN_SHARD_COUNT: usize = 1;
pub const SHARD_FILE_PREFIX: &str = FINEWEB_SHARD_FILE_PREFIX;
pub const TOKENIZATION_MARKER: &str = FINEWEB_TOKENIZATION_MARKER;

const COMPRESSION_SAMPLE_DOCUMENTS: usize = 4_096;

// The final sample-10BT parquet contains enough text for the two 25M-token
// shards needed here while avoiding a 2.15GB download for this local baseline.
const PARQUET_FILE: &str = "sample/10BT/014_00000.parquet";

pub fn parse_data() -> AppResult<()> {
    parse_data_for_train_shards(DEFAULT_TRAIN_SHARD_COUNT)
}

pub fn parse_data_for_train_shards(target_train_shards: usize) -> AppResult<()> {
    let data_dir = PathBuf::from(DATA_DIR);
    let parquet_dir = data_dir.join(PARQUET_DIR);
    let shard_dir = data_dir.join(SHARDS_DIR);
    fs::create_dir_all(&parquet_dir)?;
    fs::create_dir_all(&shard_dir)?;

    let parquet_path = download_parquet(&parquet_dir)?;
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let mut writer = ShardWriter::for_dataset(
        shard_dir.clone(),
        SHARD_FILE_PREFIX,
        SHARD_SIZE,
        target_train_shards,
    );
    let stats = tokenize_parquet(&parquet_path, &tokenizer, &mut writer)?;
    if !writer.has_required_train_and_val_shards() {
        return Err(format!(
            "{PARQUET_FILE} did not contain enough text for one validation shard and {target_train_shards} train shards"
        )
        .into());
    }
    writer.finish()?;

    fs::write(
        shard_dir.join(TOKENIZATION_MARKER),
        format!(
            "dataset={DATASET_REPO}\n\
             config={DATASET_CONFIG}\n\
             source={PARQUET_FILE}\n\
             tokenizer={TOKENIZER_NAME}\n\
             tokenizer_vocab_size={TOKENIZER_VOCAB_SIZE}\n\
             model_vocab_size={MODEL_VOCAB_SIZE}\n\
             document_boundaries={DOCUMENT_BOUNDARIES}\n\
             compression_sample_documents={}\n\
             compression_sample_bytes={}\n\
             compression_sample_tokens={}\n\
             compression_sample_bytes_per_token={:.9}\n",
            stats.sample_documents,
            stats.sample_bytes,
            stats.sample_tokens,
            stats.sample_bytes_per_token(),
        ),
    )?;
    Ok(())
}

fn download_parquet(local_dir: &Path) -> AppResult<PathBuf> {
    HFClientSync::new()?
        .dataset(DATASET_OWNER, DATASET_NAME)
        .download_file()
        .filename(PARQUET_FILE)
        .local_dir(local_dir.to_path_buf())
        .send()?;
    Ok(local_dir.join(PARQUET_FILE))
}

fn tokenize_parquet(
    path: &Path,
    tokenizer: &TrainingTokenizer,
    writer: &mut ShardWriter,
) -> AppResult<TokenizationStats> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?
        .with_batch_size(1024)
        .build()?;
    let mut stats = TokenizationStats::default();

    for batch in reader {
        let batch = batch?;
        tokenize_batch(&batch, tokenizer, writer, &mut stats)?;
        if writer.has_required_train_and_val_shards() {
            break;
        }
    }
    Ok(stats)
}

fn tokenize_batch(
    batch: &RecordBatch,
    tokenizer: &TrainingTokenizer,
    writer: &mut ShardWriter,
    stats: &mut TokenizationStats,
) -> AppResult<()> {
    let texts = string_column(batch, "text")?;
    for row in 0..batch.num_rows() {
        if let Some(text) = texts
            .get(row)
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            let token_count = tokenize_doc(text, tokenizer, writer)?;
            stats.record(text, token_count);
            if writer.has_required_train_and_val_shards() {
                break;
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct TokenizationStats {
    sample_documents: usize,
    sample_bytes: usize,
    sample_tokens: usize,
}

impl TokenizationStats {
    fn record(&mut self, text: &str, token_count: usize) {
        if self.sample_documents >= COMPRESSION_SAMPLE_DOCUMENTS {
            return;
        }
        self.sample_documents += 1;
        self.sample_bytes += text.len();
        self.sample_tokens += token_count;
    }

    fn sample_bytes_per_token(&self) -> f64 {
        self.sample_bytes as f64 / self.sample_tokens.max(1) as f64
    }
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> AppResult<TextColumn<'a>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("FineWeb parquet batch missing {name} column"))?;
    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        Ok(TextColumn::Utf8(array))
    } else if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        Ok(TextColumn::LargeUtf8(array))
    } else {
        Err(format!("FineWeb {name} column is not utf8 or large_utf8").into())
    }
}

enum TextColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl TextColumn<'_> {
    fn get(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::LargeUtf8(array) => (!array.is_null(row)).then(|| array.value(row)),
        }
    }
}
