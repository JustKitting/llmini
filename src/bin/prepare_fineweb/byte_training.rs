use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use llama2_tokenizer::TrainingTokenizer;
use sha2::{Digest, Sha256};
use synth_prep::fineweb::{SHARD_FILE_PREFIX, SHARD_SIZE, visit_parquet_text};

use crate::{
    AppResult,
    byte_reference::{MAGIC, write_document},
};

/// Match the native tokenized corpus to its raw source without detokenizing.
/// The original Parquet produced slots 0 and 1. The expansion continues at
/// slot 2 across its two Parquets, including slots reserved after tokenization.
pub(super) fn prepare(root: &Path, output: &Path, documents: usize) -> AppResult {
    let count = u32::try_from(documents)?;
    crate::reference_validation::require_tokenizer()?;
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let file = File::create_new(output)?;
    let generated = export(root, BufWriter::new(file), count, &tokenizer);
    let result = match generated {
        Ok(result) => result,
        Err(error) => {
            let _ = fs::remove_file(output);
            return Err(error);
        }
    };
    println!("byte_training_documents={}", output.display());
    println!(
        "documents={} raw_bytes={} native_tokens={} raw_sha256={:x}",
        result.documents,
        result.raw_bytes,
        result.native_tokens,
        result.raw_hash.finalize()
    );
    Ok(())
}

#[derive(Default)]
struct Stats {
    documents: u32,
    raw_bytes: u64,
    native_tokens: u64,
    raw_hash: Sha256,
}

fn export(
    root: &Path,
    mut output: impl Write + Seek,
    count: u32,
    tokenizer: &TrainingTokenizer,
) -> AppResult<Stats> {
    output.write_all(MAGIC)?;
    output.write_all(&0_u32.to_le_bytes())?;
    let shard_root = root.join("shards");
    let groups = [
        (Some(2), &["parquet/sample/10BT/014_00000.parquet"][..]),
        (
            None,
            &["parquet/000_00000.parquet", "parquet/001_00000.parquet"][..],
        ),
    ];
    let mut replay = Replay::new(&shard_root);
    let mut stats = Stats::default();
    for (end_slot, paths) in groups {
        for path in paths {
            if stats.documents == count {
                break;
            }
            visit_parquet_text(&root.join(path), |text| {
                let ids = tokenizer.encode_document(text)?;
                let belongs_to_training = replay.compare_document(&ids, end_slot)?;
                if belongs_to_training {
                    let raw = text.as_bytes();
                    write_document(&mut output, raw, true)?;
                    stats.documents += 1;
                    stats.raw_bytes += raw.len() as u64;
                    stats.native_tokens += ids.len() as u64;
                    stats.raw_hash.update(raw);
                }
                Ok(stats.documents < count && end_slot != Some(replay.slot))
            })?;
        }
        if stats.documents == count {
            break;
        }
        if let Some(expected) = end_slot
            && replay.slot != expected
        {
            return Err(format!(
                "raw source ended before completing native slot {}",
                expected - 1
            )
            .into());
        }
    }
    if stats.documents != count {
        return Err(format!(
            "replayed {} of {count} requested training documents",
            stats.documents
        )
        .into());
    }
    output.seek(SeekFrom::Start(MAGIC.len() as u64))?;
    output.write_all(&count.to_le_bytes())?;
    output.flush()?;
    Ok(stats)
}

struct Replay<'a> {
    shard_root: &'a Path,
    slot: usize,
    offset: usize,
    reader: Option<BufReader<File>>,
    training: bool,
    bytes: Vec<u8>,
}

impl<'a> Replay<'a> {
    fn new(shard_root: &'a Path) -> Self {
        Self {
            shard_root,
            slot: 0,
            offset: 0,
            reader: None,
            training: false,
            bytes: Vec::new(),
        }
    }

    fn compare_document(&mut self, mut ids: &[u32], end_slot: Option<usize>) -> AppResult<bool> {
        let mut training = true;
        while !ids.is_empty() {
            if end_slot == Some(self.slot) {
                return Ok(false);
            }
            if self.reader.is_none() {
                self.open_slot()?;
            }
            training &= self.training;
            let length = (SHARD_SIZE - self.offset).min(ids.len());
            self.bytes.resize(length * 2, 0);
            self.reader
                .as_mut()
                .expect("open replay slot")
                .read_exact(&mut self.bytes)?;
            for (position, (&expected, actual)) in ids[..length]
                .iter()
                .zip(self.bytes.chunks_exact(2))
                .enumerate()
            {
                let actual = u16::from_ne_bytes([actual[0], actual[1]]);
                if expected != u32::from(actual) {
                    return Err(format!("FineWeb raw replay differs at native shard {} token {} (source {expected}, stored {actual})", self.slot, self.offset + position).into());
                }
            }
            ids = &ids[length..];
            self.offset += length;
            if self.offset == SHARD_SIZE {
                self.slot += 1;
                self.offset = 0;
                self.reader = None;
            }
        }
        Ok(training)
    }

    fn open_slot(&mut self) -> AppResult {
        let kind = if self.slot == 0 { "val" } else { "train" };
        let canonical = self
            .shard_root
            .join(format!("{SHARD_FILE_PREFIX}_{kind}_{:06}.bin", self.slot));
        let file = match File::open(&canonical) {
            Ok(file) => {
                self.training = self.slot != 0;
                file
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && self.slot != 0 => {
                self.training = false;
                File::open(self.shard_root.join(format!(
                    "{SHARD_FILE_PREFIX}_reserved_nanogpt_{:06}.bin",
                    self.slot
                )))?
            }
            Err(error) => return Err(error.into()),
        };
        if file.metadata()?.len() != (2 * SHARD_SIZE) as u64 {
            return Err(format!("incomplete native FineWeb replay shard {}", self.slot).into());
        }
        self.reader = Some(BufReader::new(file));
        Ok(())
    }
}
