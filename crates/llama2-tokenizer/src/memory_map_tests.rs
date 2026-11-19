//! Reference addressing contracts written before the tokenizer fidelity repair.
//! These exercise the actual tokenizer assets; no model training occurs.
use super::*;

fn id(tokenizer: &TrainingTokenizer, spelling: &str) -> usize {
    tokenizer.tokenizer.token_to_id(spelling).unwrap() as usize
}

#[test]
fn reference_memory_map_merges_valid_ascii_byte_spellings() -> AppResult<()> {
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let map = tokenizer.memory_id_map()?;
    let ids = ["<0x41>", "<0x61>", "A", "a"].map(|spelling| id(&tokenizer, spelling));
    for token in ids {
        let decoded = tokenizer
            .tokenizer
            .decode(&[token as u32], false)
            .map_err(tokenizer_error)?;
        assert!(
            decoded == "A" || decoded == "a",
            "unexpected fixture token {token}: {decoded:?}"
        );
        assert_eq!(
            map[token], map[ids[0]],
            "valid ASCII bytes must follow the reference normalizer, not the replacement-character fallback"
        );
    }
    println!("reference memory classes={}", map.iter().max().unwrap() + 1);
    Ok(())
}

#[test]
fn reference_memory_map_merges_decoded_tab_and_newline() -> AppResult<()> {
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let map = tokenizer.memory_id_map()?;
    let tab = id(&tokenizer, "<0x09>");
    let newline = id(&tokenizer, "<0x0A>");
    for token in [tab, newline] {
        let decoded = tokenizer
            .tokenizer
            .decode(&[token as u32], false)
            .map_err(tokenizer_error)?;
        assert!(
            decoded == "\t" || decoded == "\n",
            "unexpected fixture {decoded:?}"
        );
    }
    assert_eq!(
        map[tab], map[newline],
        "the reference replaces decoded ASCII whitespace runs before stripping"
    );
    Ok(())
}

#[test]
fn reference_memory_map_retains_invalid_utf8_and_special_distinctions() -> AppResult<()> {
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let map = tokenizer.memory_id_map()?;
    let invalid = ["<0x80>", "<0x81>"].map(|spelling| id(&tokenizer, spelling));
    for token in invalid {
        let decoded = tokenizer
            .tokenizer
            .decode(&[token as u32], false)
            .map_err(tokenizer_error)?;
        assert!(decoded.contains('\u{fffd}'));
    }
    assert_ne!(map[invalid[0]], map[invalid[1]]);
    assert_ne!(map[BOS_TOKEN as usize], map[EOS_TOKEN as usize]);
    assert_ne!(map[0], map[BOS_TOKEN as usize]);
    let encoded = tokenizer.encode_document("ASCII A/a, markup <DIV>, café\tline\nnext.")?;
    assert_eq!(map, tokenizer.memory_id_map()?);
    assert_eq!(
        encoded,
        tokenizer.encode_document("ASCII A/a, markup <DIV>, café\tline\nnext.")?
    );
    Ok(())
}
