//! Exact normalization/address assignment from the pinned Engram demo.
//! This map addresses memory only; LM tokens, targets and decoding stay intact.
use super::{AppResult, MODEL_VOCAB_SIZE, tokenizer_error};
use std::collections::HashMap;
use tokenizers::{
    NormalizedString, Normalizer, Tokenizer,
    normalizers::{
        Lowercase, NFD, NFKC, Replace, Sequence, Strip, StripAccents, replace::ReplacePattern,
    },
};

pub(super) fn id_map(tokenizer: &Tokenizer) -> AppResult<Vec<u32>> {
    // The reference's private-use sentinel preserves a pure ASCII-space token
    // across Strip. Keep its exact ordering, whitespace class and empty fallback.
    const SENTINEL: &str = "\u{e000}";
    let normalizer = Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")
            .map_err(tokenizer_error)?
            .into(),
        Replace::new(ReplacePattern::Regex(r"^ $".into()), SENTINEL)
            .map_err(tokenizer_error)?
            .into(),
        Strip::new(true, true).into(),
        Replace::new(SENTINEL, " ").map_err(tokenizer_error)?.into(),
    ]);
    if tokenizer.get_vocab_size(true) != MODEL_VOCAB_SIZE {
        return Err("Engram memory map requires the complete selected tokenizer vocabulary".into());
    }
    let mut keys = HashMap::<String, u32>::new();
    let mut map = Vec::with_capacity(MODEL_VOCAB_SIZE);
    for id in 0..MODEL_VOCAB_SIZE as u32 {
        let text = tokenizer.decode(&[id], false).map_err(tokenizer_error)?;
        let key = if text.contains('\u{fffd}') {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| format!("missing tokenizer spelling for memory ID {id}"))?
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(tokenizer_error)?;
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next = keys.len() as u32;
        map.push(*keys.entry(key).or_insert(next));
    }
    Ok(map)
}
