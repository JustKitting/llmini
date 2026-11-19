use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

use burn::serde::{Deserialize, Serialize};
use gpt2_nvfp4::{GPT2_BATCH_SIZE, GPT2_SEQ_LEN};
use serde_json::json;

use crate::AppResult;

const TOKEN_BYTES: usize = size_of::<u16>();
const ROW_BYTES: usize = (GPT2_SEQ_LEN + 1) * TOKEN_BYTES;
const CURATED_ROWS: [usize; 3] = [2, 3, 3];
const ROTATION: usize = 5;

#[derive(Deserialize, Serialize)]
#[serde(crate = "burn::serde")]
struct Manifest {
    tokenizer: String,
    shards: Vec<PathBuf>,
}

struct Shard {
    path: PathBuf,
    batches: usize,
    independent: bool,
}

impl Shard {
    fn read_layout(path: PathBuf) -> AppResult<Self> {
        let independent = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".windows.bin"));
        let meta = path.metadata()?;
        let len = usize::try_from(meta.len())?;
        let batches = if independent {
            len / (GPT2_BATCH_SIZE * ROW_BYTES)
        } else {
            (len / TOKEN_BYTES).saturating_sub(1) / (GPT2_BATCH_SIZE * GPT2_SEQ_LEN)
        };
        if !meta.is_file()
            || batches == 0
            || !len.is_multiple_of(TOKEN_BYTES)
            || (independent && !len.is_multiple_of(GPT2_BATCH_SIZE * ROW_BYTES))
        {
            return Err(format!("invalid native training shard {}", path.display()).into());
        }
        Ok(Self {
            path,
            batches,
            independent,
        })
    }
}

struct ShardReader {
    file: BufReader<File>,
    remaining: usize,
    overlap: bool,
    last_target: Option<[u8; TOKEN_BYTES]>,
}

impl ShardReader {
    fn read_row(&mut self, row: &mut [u8; ROW_BYTES]) -> AppResult {
        let offset = if let Some(target) = self.last_target {
            row[..TOKEN_BYTES].copy_from_slice(&target);
            TOKEN_BYTES
        } else {
            0
        };
        self.file.read_exact(&mut row[offset..])?;
        if self.overlap {
            self.last_target = Some(row[ROW_BYTES - TOKEN_BYTES..].try_into()?);
        }
        self.remaining -= 1;
        Ok(())
    }
}

struct Pool<'a> {
    shards: std::vec::IntoIter<&'a Shard>,
    current: Option<ShardReader>,
}

impl<'a> Pool<'a> {
    fn new(shards: Vec<&'a Shard>) -> Self {
        Self {
            shards: shards.into_iter(),
            current: None,
        }
    }

    fn read_row(&mut self, row: &mut [u8; ROW_BYTES]) -> AppResult {
        if self
            .current
            .as_ref()
            .is_none_or(|shard| shard.remaining == 0)
        {
            let shard = self
                .shards
                .next()
                .ok_or("interleave source pool exhausted")?;
            self.current = Some(ShardReader {
                file: BufReader::with_capacity(1024 * 1024, File::open(&shard.path)?),
                remaining: shard.batches * GPT2_BATCH_SIZE,
                overlap: !shard.independent,
                last_target: None,
            });
        }
        self.current.as_mut().unwrap().read_row(row)
    }
}

pub(super) fn prepare(source: &Path, output: &Path, curated_shards: &[NonZeroUsize]) -> AppResult {
    if GPT2_BATCH_SIZE != 8 {
        return Err("the matched 2,3,3 row schedule uses an eight-row batch".into());
    }
    let manifest: Manifest = serde_json::from_reader(BufReader::new(File::open(source)?))?;
    if manifest.tokenizer != llama2_tokenizer::TOKENIZER_NAME {
        return Err(format!(
            "manifest uses tokenizer {}, expected {}",
            manifest.tokenizer,
            llama2_tokenizer::TOKENIZER_NAME,
        )
        .into());
    }
    let mut curated = vec![false; manifest.shards.len()];
    for &index in curated_shards {
        let flag = curated
            .get_mut(index.get() - 1)
            .ok_or("curated shard index is out of range")?;
        if std::mem::replace(flag, true) {
            return Err("curated shard indices must be distinct".into());
        }
    }
    let first = curated
        .iter()
        .position(|&item| item)
        .ok_or("no curated shards")?;
    let end = curated.iter().rposition(|&item| item).unwrap() + 1;
    let source_root = parent(source).canonicalize()?;
    let shards: Vec<_> = manifest
        .shards
        .iter()
        .map(|path| Shard::read_layout(source_root.join(path)))
        .collect::<AppResult<_>>()?;

    let middle = &shards[first..end];
    let middle_batches: usize = middle.iter().map(|shard| shard.batches).sum();
    let curated_batches: usize = middle
        .iter()
        .zip(&curated[first..end])
        .filter_map(|(shard, &flag)| flag.then_some(shard.batches))
        .sum();
    if !middle_batches.is_multiple_of(3) || curated_batches != middle_batches / 3 {
        return Err(
            "the middle region must supply one third curated rows in whole three-batch groups"
                .into(),
        );
    }
    let (curated_pool, fineweb_pool) = middle
        .iter()
        .zip(&curated[first..end])
        .partition::<Vec<_>, _>(|(_, flag)| **flag);
    let mut curated_pool = Pool::new(curated_pool.into_iter().map(|(shard, _)| shard).collect());
    let mut fineweb_pool = Pool::new(fineweb_pool.into_iter().map(|(shard, _)| shard).collect());

    let output_name = output
        .file_name()
        .ok_or("expected a new output directory")?;
    let output = parent(output).canonicalize()?.join(output_name);
    let source_relative = relative_to(&source_root, &output);
    let mut names: Vec<_> = manifest.shards[..first]
        .iter()
        .map(|path| source_relative.join(path))
        .collect();
    fs::create_dir(&output)?;
    let mut row = [0; ROW_BYTES];
    let mut batch = 0;
    for (index, shard) in middle.iter().enumerate() {
        let name = PathBuf::from(format!("interleaved-{:02}.windows.bin", index + 1));
        let mut writer =
            BufWriter::with_capacity(1024 * 1024, File::create_new(output.join(&name))?);
        for _ in 0..shard.batches {
            let count = CURATED_ROWS[batch % CURATED_ROWS.len()];
            let rotate = (batch % GPT2_BATCH_SIZE) * ROTATION % GPT2_BATCH_SIZE;
            for slot in 0..GPT2_BATCH_SIZE {
                let pool = if (slot + GPT2_BATCH_SIZE - rotate) % GPT2_BATCH_SIZE < count {
                    &mut curated_pool
                } else {
                    &mut fineweb_pool
                };
                pool.read_row(&mut row)?;
                writer.write_all(&row)?;
            }
            batch += 1;
        }
        writer.flush()?;
        names.push(name);
    }
    names.extend(
        manifest.shards[end..]
            .iter()
            .map(|path| source_relative.join(path)),
    );
    let training_manifest = output.join("training.json");
    let mut file = File::create_new(&training_manifest)?;
    serde_json::to_writer_pretty(
        &mut file,
        &Manifest {
            tokenizer: manifest.tokenizer,
            shards: names,
        },
    )?;
    file.write_all(b"\n")?;

    let prefix: usize = shards[..first].iter().map(|shard| shard.batches).sum();
    let curated_rows = curated_batches * GPT2_BATCH_SIZE;
    println!(
        "{}",
        json!({
            "source_manifest": source,
            "training_manifest": training_manifest,
            "batch_size": GPT2_BATCH_SIZE,
            "sequence_length": GPT2_SEQ_LEN,
            "first_mixed_attempt": prefix + 1,
            "last_mixed_attempt": prefix + middle_batches,
            "curated_rows": curated_rows,
            "fineweb_rows": middle_batches * GPT2_BATCH_SIZE - curated_rows,
            "curated_input_tokens": curated_rows * GPT2_SEQ_LEN,
        })
    );
    Ok(())
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn relative_to(path: &Path, base: &Path) -> PathBuf {
    let common = base
        .ancestors()
        .find(|ancestor| path.starts_with(ancestor))
        .unwrap();
    let mut relative = PathBuf::new();
    relative.extend(std::iter::repeat_n(
        "..",
        base.strip_prefix(common).unwrap().components().count(),
    ));
    relative.push(path.strip_prefix(common).unwrap());
    relative
}
