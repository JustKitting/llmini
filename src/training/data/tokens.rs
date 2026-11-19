use std::{fs, io, path::Path};

use crate::AppResult;

pub(super) fn read_u16_tokens(path: &Path) -> AppResult<Vec<u16>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} has odd byte length", path.display()).into());
    }

    let tokens = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    if let Some((offset, id)) = tokens
        .iter()
        .enumerate()
        .find(|&(_, &id)| usize::from(id) >= llama2_tokenizer::MODEL_VOCAB_SIZE)
    {
        return Err(format!(
            "{} has token ID {id} outside the model vocabulary at offset {offset}",
            path.display()
        )
        .into());
    }
    Ok(tokens)
}

pub(super) fn write_u16_tokens(path: &Path, tokens: &[u16]) -> AppResult<()> {
    let bytes = tokens
        .iter()
        .flat_map(|&token| token.to_ne_bytes())
        .collect::<Vec<_>>();
    fs::write(path, bytes).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("failed to write {}: {err}", path.display()),
        )
        .into()
    })
}
