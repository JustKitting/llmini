//! Read the same overlapping token windows as the native training loader.
use crate::AppResult;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub(super) struct InputWindow {
    pub tokens: Vec<u16>,
    pub metadata: Value,
}

pub(super) fn load(
    shard: &Path,
    offset: usize,
    batch_size: usize,
    seq_len: usize,
) -> AppResult<InputWindow> {
    if batch_size == 0 || seq_len == 0 {
        return Err("gradient window requires nonzero batch and sequence lengths".into());
    }
    let source_tokens = batch_size
        .checked_mul(seq_len)
        .and_then(|n| n.checked_add(1))
        .ok_or("gradient window token count overflow")?;
    let flat_tokens = seq_len
        .checked_add(1)
        .and_then(|n| batch_size.checked_mul(n))
        .ok_or("gradient window flattened length overflow")?;
    let byte_offset = offset
        .checked_mul(2)
        .ok_or("gradient window offset overflow")?;
    let byte_len = source_tokens
        .checked_mul(2)
        .ok_or("gradient window byte length overflow")?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or("gradient window end overflow")?;
    let mut file = File::open(shard)?;
    let file_len = file.metadata()?.len();
    if file_len % 2 != 0 {
        return Err(format!("{} has odd byte length", shard.display()).into());
    }
    if u64::try_from(byte_end)? > file_len {
        return Err(format!(
            "gradient window at token {offset} requires {source_tokens} tokens beyond {}",
            shard.display()
        )
        .into());
    }
    file.seek(SeekFrom::Start(u64::try_from(byte_offset)?))?;
    let mut bytes = vec![0; byte_len];
    file.read_exact(&mut bytes)?;
    let ids = bytes
        .chunks_exact(2)
        .map(|v| u16::from_ne_bytes([v[0], v[1]]))
        .collect::<Vec<_>>();
    if ids
        .iter()
        .any(|&v| usize::from(v) >= llama2_tokenizer::MODEL_VOCAB_SIZE)
    {
        return Err(
            "gradient window contains a token outside the compiled model vocabulary".into(),
        );
    }
    let mut tokens = Vec::with_capacity(flat_tokens);
    for sequence in 0..batch_size {
        let start = sequence * seq_len;
        tokens.extend_from_slice(&ids[start..start + seq_len + 1]);
    }
    Ok(InputWindow {
        tokens,
        metadata: json!({"source":shard,"token_offset":offset,"batch_size":batch_size,
            "seq_len":seq_len,"source_tokens":source_tokens,
            "source_window_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "layout":"batch_size adjacent seq_len-token sequences, each with its following target token"}),
    })
}
