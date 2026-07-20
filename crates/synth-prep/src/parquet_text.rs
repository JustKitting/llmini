use std::fs::File;
use std::path::Path;

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use llama2_tokenizer::TrainingTokenizer;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::AppResult;
use super::shards::ShardWriter;
use super::tokenize::tokenize_doc;

pub fn tokenize_parquet_file(
    path: &Path,
    tokenizer: &TrainingTokenizer,
    writer: &mut ShardWriter,
) -> AppResult<()> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(1024)
        .build()?;

    for batch in reader {
        if writer.has_required_train_and_val_shards() {
            break;
        }
        let batch = batch?;
        tokenize_synth_batch(&batch, tokenizer, writer)?;
    }

    Ok(())
}

fn tokenize_synth_batch(
    batch: &RecordBatch,
    tokenizer: &TrainingTokenizer,
    writer: &mut ShardWriter,
) -> AppResult<()> {
    let sections = [
        ("Query", string_column(batch, "query")?),
        ("Reasoning", string_column(batch, "synthetic_reasoning")?),
        ("Answer", string_column(batch, "synthetic_answer")?),
    ];

    for row in 0..batch.num_rows() {
        let mut text = String::new();
        for (label, column) in &sections {
            append_section(&mut text, label, column.get(row));
        }

        if !text.is_empty() {
            tokenize_doc(&text, tokenizer, writer)?;
            if writer.has_required_train_and_val_shards() {
                break;
            }
        }
    }
    Ok(())
}

fn append_section(text: &mut String, label: &str, value: Option<&str>) {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return;
    };

    if !text.is_empty() {
        text.push_str("\n\n");
    }
    text.push_str(label);
    text.push_str(":\n");
    text.push_str(value);
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> AppResult<SynthColumn<'a>> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("SYNTH parquet batch missing {name} column"))?;

    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        Ok(SynthColumn::Utf8(array))
    } else if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        Ok(SynthColumn::LargeUtf8(array))
    } else {
        Err(format!("SYNTH {name} column is not utf8 or large_utf8").into())
    }
}

enum SynthColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl SynthColumn<'_> {
    fn get(&self, row: usize) -> Option<&str> {
        match self {
            Self::Utf8(array) => (!array.is_null(row)).then(|| array.value(row)),
            Self::LargeUtf8(array) => (!array.is_null(row)).then(|| array.value(row)),
        }
    }
}
