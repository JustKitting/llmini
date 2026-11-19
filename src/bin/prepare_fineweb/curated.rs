use crate::{AppResult, reference_validation as reference};
use llama2_tokenizer::{MODEL_VOCAB_SIZE, TOKENIZER_NAME, TrainingTokenizer};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

const GRAM: usize = 13;
const BASE: u64 = 1_099_511_628_211;
const LEADING: u64 = BASE.wrapping_pow((GRAM - 1) as u32);

pub fn prepare(
    parquet: &Path,
    output: &Path,
    limit: usize,
    documents: &Path,
    answers: &[PathBuf],
) -> AppResult {
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let source_sha256 = reference::sha256(parquet)?;
    let mut reserved = HashSet::with_capacity(reference::TARGETS);
    let bos = u16::try_from(tokenizer.bos_token())?;
    let eos = u16::try_from(tokenizer.eos_token())?;
    reference::visit_documents(documents, |ids| {
        let tokens = &ids[1..];
        reserved.extend(fingerprints(tokens.strip_suffix(&[eos]).unwrap_or(tokens)));
        Ok(())
    })?;
    let mut answer_reports = Vec::new();
    for path in answers {
        answer_reports.push(index_answers(path, &tokenizer, &mut reserved)?);
    }

    let file = File::create_new(output)?;
    let stats = match write_stream(parquet, file, limit, &tokenizer, bos, eos, &reserved) {
        Ok(stats) => stats,
        Err(error) => {
            fs::remove_file(output)?;
            return Err(error);
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source": parquet.file_name().map(|name| name.to_string_lossy()),
            "source_sha256": source_sha256,
            "output_sha256": reference::sha256(output)?,
            "encoding": "u16-le",
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_sha256": reference::TOKENIZER_SHA,
            "tokens": stats.tokens,
            "documents_examined": stats.examined,
            "documents_written": stats.written,
            "documents_excluded_overlap": stats.overlap,
            "documents_excluded_exact_duplicate": stats.duplicate,
            "screen_ngram_tokens": GRAM,
            "screen_unique_fingerprints": reserved.len(),
            "reserved_documents_sha256": reference::DATA_SHA,
            "reserved_documents": reference::DOCUMENTS,
            "reserved_answers": answer_reports,
        }))?
    );
    Ok(())
}

#[derive(Default)]
struct Stats {
    tokens: usize,
    examined: usize,
    written: usize,
    overlap: usize,
    duplicate: usize,
}

fn write_stream(
    parquet: &Path,
    file: File,
    limit: usize,
    tokenizer: &TrainingTokenizer,
    bos: u16,
    eos: u16,
    reserved: &HashSet<u64>,
) -> AppResult<Stats> {
    let mut writer = BufWriter::new(file);
    let mut stats = Stats::default();
    let mut seen = HashSet::<[u8; 32]>::new();
    synth_prep::fineweb::visit_parquet_text(parquet, |text| {
        stats.examined += 1;
        if !seen.insert(Sha256::digest(text.as_bytes()).into()) {
            stats.duplicate += 1;
            return Ok(true);
        }
        let ids = token_ids(tokenizer.encode_document(text)?)?;
        let content = ids
            .strip_prefix(&[bos])
            .and_then(|ids| ids.strip_suffix(&[eos]))
            .ok_or("curated document is missing the tokenizer boundaries")?;
        if fingerprints(content).any(|gram| reserved.contains(&gram)) {
            stats.overlap += 1;
            return Ok(true);
        }
        let count = ids.len().min(limit - stats.tokens);
        for &id in &ids[..count] {
            writer.write_all(&id.to_le_bytes())?;
        }
        stats.tokens += count;
        stats.written += 1;
        Ok(stats.tokens < limit)
    })?;
    if stats.tokens != limit {
        return Err(format!(
            "{}: only {} tokens after screening; requested {limit}",
            parquet.display(),
            stats.tokens
        )
        .into());
    }
    writer.flush()?;
    Ok(stats)
}

fn index_answers(
    path: &Path,
    tokenizer: &TrainingTokenizer,
    reserved: &mut HashSet<u64>,
) -> AppResult<Value> {
    let mut count = 0;
    for (line, text) in BufReader::new(File::open(path)?).lines().enumerate() {
        let location = || format!("{}:{}", path.display(), line + 1);
        let record: Value =
            serde_json::from_str(&text?).map_err(|error| format!("{}: {error}", location()))?;
        let field = |name| {
            record[name]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| format!("{}: expected nonempty {name}", location()))
        };
        let prompt = field("prompt")?;
        let answer = field("answer")?;
        for text in [answer, &format!("{prompt}{answer}")] {
            let ids = token_ids(tokenizer.encode_ordinary(text)?)?;
            reserved.extend(fingerprints(&ids));
        }
        count += 1;
    }
    if count == 0 {
        return Err(format!("{}: empty reserved answer set", path.display()).into());
    }
    Ok(json!({
        "source": path.file_name().map(|name| name.to_string_lossy()),
        "sha256": reference::sha256(path)?,
        "records": count,
    }))
}

fn token_ids(ids: Vec<u32>) -> AppResult<Vec<u16>> {
    ids.into_iter()
        .map(|id| {
            if id as usize >= MODEL_VOCAB_SIZE {
                return Err(format!("token {id} exceeds the model vocabulary").into());
            }
            Ok(u16::try_from(id)?)
        })
        .collect()
}

fn fingerprints(tokens: &[u16]) -> impl Iterator<Item = u64> + '_ {
    let first = tokens.get(..GRAM).map(|ids| {
        ids.iter().fold(0_u64, |hash, &id| {
            hash.wrapping_mul(BASE).wrapping_add(u64::from(id) + 1)
        })
    });
    let tail = tokens.iter().zip(tokens.iter().skip(GRAM)).scan(
        first.unwrap_or_default(),
        |hash, (&outgoing, &incoming)| {
            *hash = hash
                .wrapping_sub((u64::from(outgoing) + 1).wrapping_mul(LEADING))
                .wrapping_mul(BASE)
                .wrapping_add(u64::from(incoming) + 1);
            Some(*hash)
        },
    );
    first.into_iter().chain(tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_fingerprints_match_independent_windows() {
        let tokens = (0..129)
            .map(|i| (i * 997 % MODEL_VOCAB_SIZE) as u16)
            .collect::<Vec<_>>();
        for length in [0, GRAM - 1, GRAM, GRAM + 1, tokens.len()] {
            let slice = &tokens[..length];
            let expected = slice.windows(GRAM).map(|gram| {
                gram.iter()
                    .zip((0..GRAM).rev())
                    .fold(0_u64, |hash, (&id, power)| {
                        hash.wrapping_add(
                            BASE.wrapping_pow(power as u32)
                                .wrapping_mul(u64::from(id) + 1),
                        )
                    })
            });
            assert!(fingerprints(slice).eq(expected));
        }
        let reserved = fingerprints(&tokens[19..19 + GRAM]).collect::<HashSet<_>>();
        assert!(fingerprints(&tokens).any(|gram| reserved.contains(&gram)));
    }
}
