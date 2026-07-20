use std::fs;
use std::path::{Path, PathBuf};

use llama2_tokenizer::{TOKENIZER_NAME, TrainingTokenizer};
use synth_prep::synth::SHARDS_DIR;

use super::tokens::{read_u16_tokens, write_u16_tokens};
use crate::AppResult;

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
const SHAKESPEARE_DIR: &str = "data/shakespeare";
const SHAKESPEARE_RAW: &str = "input.txt";

pub(super) fn ensure_shard() -> AppResult<PathBuf> {
    let dir = Path::new(SHAKESPEARE_DIR);
    let shard_dir = dir.join(SHARDS_DIR);
    let shard_path = shard_dir.join(format!("shakespeare_{TOKENIZER_NAME}_train_000000.bin"));
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    if shard_path.exists() && shard_contains_token(&shard_path, tokenizer.eos_token())? {
        return Ok(shard_path);
    }

    fs::create_dir_all(&shard_dir)?;
    let raw_path = dir.join(SHAKESPEARE_RAW);
    if !raw_path.exists() {
        let text = reqwest::blocking::get(SHAKESPEARE_URL)?
            .error_for_status()?
            .text()?;
        fs::create_dir_all(dir)?;
        fs::write(&raw_path, text)?;
    }

    let text = fs::read_to_string(raw_path)?;
    let tokens = tokenizer
        .encode_document(&text)?
        .into_iter()
        .map(u16::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    write_u16_tokens(&shard_path, &tokens)?;
    Ok(shard_path)
}

fn shard_contains_token(path: &Path, token: u32) -> AppResult<bool> {
    let token = u16::try_from(token)?;
    Ok(read_u16_tokens(path)?.contains(&token))
}
