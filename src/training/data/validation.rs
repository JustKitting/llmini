use std::path::Path;

use gpt2_nvfp4::GPT2_SEQ_LEN;
use llama2_tokenizer::TrainingTokenizer;

use crate::AppResult;

pub(in crate::training) const VALIDATION_WINDOWS: usize = 4;

pub(in crate::training) fn validation_target_byte_count(tokens: &[u16]) -> AppResult<usize> {
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let window_len = GPT2_SEQ_LEN + 1;
    let needed = VALIDATION_WINDOWS * window_len;
    if tokens.len() < needed {
        return Err(format!(
            "validation byte count has {} tokens, needs {needed}",
            tokens.len()
        )
        .into());
    }

    let mut byte_count = 0usize;
    for window in tokens[..needed].chunks_exact(window_len) {
        let target_ids = window[1..]
            .iter()
            .map(|&token| u32::from(token))
            .collect::<Vec<_>>();
        byte_count += tokenizer.decode(&target_ids)?.len();
    }
    if byte_count == 0 {
        return Err("validation targets decode to zero bytes".into());
    }
    Ok(byte_count)
}

pub(super) fn validation_windows(path: &Path, tokens: &[u16], start: usize) -> AppResult<Vec<u16>> {
    let len = GPT2_SEQ_LEN + 1;
    if tokens.len() < len {
        return Err(format!("{} has fewer than {len} tokens", path.display()).into());
    }

    let needed = VALIDATION_WINDOWS * len;
    if tokens.len() < start + needed {
        return Err(format!(
            "{} has fewer than {} validation tokens",
            path.display(),
            start + needed
        )
        .into());
    }

    Ok(tokens[start..start + needed].to_vec())
}

pub(super) fn train_end(token_count: usize) -> usize {
    token_count
        .saturating_sub(VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1))
        .max(GPT2_SEQ_LEN + 1)
}

#[cfg(test)]
mod tests {
    use super::{VALIDATION_WINDOWS, validation_target_byte_count};
    use gpt2_nvfp4::GPT2_SEQ_LEN;
    use llama2_tokenizer::TrainingTokenizer;

    #[test]
    fn validation_byte_count_decodes_each_target_window() {
        let tokenizer = TrainingTokenizer::from_default_assets().unwrap();
        let text = "The tokenizer-normalized validation metric counts decoded UTF-8 bytes. ";
        let encoded = tokenizer.encode_document(text).unwrap();
        let needed = VALIDATION_WINDOWS * (GPT2_SEQ_LEN + 1);
        let tokens = encoded
            .into_iter()
            .cycle()
            .take(needed)
            .map(|token| u16::try_from(token).unwrap())
            .collect::<Vec<_>>();
        assert!(validation_target_byte_count(&tokens).unwrap() > 0);
    }
}
