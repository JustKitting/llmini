use crate::{AppResult, reference_validation as reference};
use llama2_tokenizer::TrainingTokenizer;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

pub fn prepare(source: &Path, vocab: &Path, output: &Path) -> AppResult {
    reference::require_tokenizer()?;
    reference::require_hash(source, reference::SOURCE_SHA)?;
    reference::require_hash(vocab, reference::VOCAB_SHA)?;
    let raw_by_id = decode_vocabulary(vocab)?;
    let mut source = BufReader::new(File::open(source)?);
    let mut header = [0_u8; 1024];
    source.read_exact(&mut header)?;
    let word = |n: usize| u32::from_le_bytes(header[n * 4..n * 4 + 4].try_into().unwrap());
    if word(0) != 20240520 || word(1) != 1 || word(2) < (reference::SOURCE_TARGETS + 1) as u32 {
        return Err("invalid official FineWeb token shard header".into());
    }
    let mut raw_ids = vec![0_u8; 2 * (reference::SOURCE_TARGETS + 1)];
    source.read_exact(&mut raw_ids)?;
    let ids = raw_ids
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect::<Vec<_>>();
    if ids[0] != 50256 {
        return Err("official prefix must start with EOT".into());
    }
    let mut spans = Vec::new();
    let mut start = 1;
    for (index, &token) in ids.iter().enumerate().skip(1) {
        if token == 50256 {
            spans.push((start, index, true));
            start = index + 1;
        }
    }
    if start < ids.len() {
        spans.push((start, ids.len(), false));
    }
    if spans.len() != reference::DOCUMENTS {
        return Err("reference document count changed".into());
    }

    fs::create_dir_all(output)?;
    let partial = output.join("documents.bin.partial");
    let destination = output.join("documents.bin");
    let mut writer = BufWriter::new(File::create(&partial)?);
    writer.write_all(reference::MAGIC)?;
    write_u32(&mut writer, spans.len())?;
    write_u32(&mut writer, reference::CONTEXT)?;
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let mut raw_hash = Sha256::new();
    let mut raw_bytes = 0;
    let mut targets = 0;
    for (document, &(lo, hi, closed)) in spans.iter().enumerate() {
        let mut raw = Vec::new();
        for &id in &ids[lo..hi] {
            raw.extend_from_slice(raw_by_id.get(id as usize).ok_or("invalid GPT-2 token")?);
        }
        let text = std::str::from_utf8(&raw)?;
        let ordinary = tokenizer.encode_ordinary(text)?;
        if tokenizer.decode(&ordinary)?.as_bytes() != raw {
            return Err(format!("tokenizer changed bytes in reference document {document}").into());
        }
        let mut tokens = Vec::with_capacity(ordinary.len() + 2);
        tokens.push(1_u16);
        for token in ordinary {
            if token >= 32000 {
                return Err("reference token is outside model vocabulary".into());
            }
            tokens.push(token as u16);
        }
        if closed {
            tokens.push(2);
        }
        write_u32(&mut writer, document)?;
        write_u32(&mut writer, raw.len())?;
        write_u32(&mut writer, usize::from(closed))?;
        write_u32(&mut writer, tokens.len())?;
        for token in tokens.iter() {
            writer.write_all(&token.to_le_bytes())?;
        }
        raw_hash.update(&raw);
        raw_bytes += raw.len();
        targets += tokens.len() - 1;
        if (document + 1) % 2000 == 0 {
            println!(
                "prepared_reference_documents={}/{}",
                document + 1,
                spans.len()
            );
        }
    }
    writer.flush()?;
    drop(writer);
    let raw_sha = format!("{:x}", raw_hash.finalize());
    if raw_bytes != reference::RAW_BYTES
        || targets != reference::TARGETS
        || raw_sha != reference::RAW_SHA
    {
        return Err("reference byte/target accounting changed".into());
    }
    reference::require_hash(&partial, reference::DATA_SHA)?;
    fs::rename(partial, &destination)?;
    let metadata = serde_json::json!({
        "status": "prepared",
        "source_url": "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin",
        "source_sha256": reference::SOURCE_SHA,
        "vocab_sha256": reference::VOCAB_SHA,
        "tokenizer_sha256": reference::TOKENIZER_SHA,
        "data_sha256": reference::DATA_SHA,
        "raw_text_sha256": raw_sha,
        "official_source_target_tokens": reference::SOURCE_TARGETS,
        "raw_target_bytes": raw_bytes,
        "documents": spans.len(),
        "scored_target_tokens": targets,
        "roundtrip_differing_documents": 0,
        "context_tokens": reference::CONTEXT,
        "sliding_stride_tokens": reference::STRIDE,
        "special_token_policy": "BOS conditions each document; score each text/EOS target once; no BOS/padding loss; no EOS on final partial document",
        "reference_target_bpb_calculated": 3.28 * reference::SOURCE_TARGETS as f64 / raw_bytes as f64 / std::f64::consts::LN_2,
    });
    fs::write(
        output.join("documents.json"),
        serde_json::to_string_pretty(&metadata)? + "\n",
    )?;
    println!("reference_documents={}", destination.display());
    println!("{metadata}");
    Ok(())
}

fn write_u32(writer: &mut impl Write, value: usize) -> AppResult {
    writer.write_all(&u32::try_from(value)?.to_le_bytes())?;
    Ok(())
}

fn decode_vocabulary(path: &Path) -> AppResult<Vec<Vec<u8>>> {
    let vocab: HashMap<String, usize> = serde_json::from_reader(File::open(path)?)?;
    let mut alphabet = HashMap::new();
    let mut present = [false; 256];
    for byte in (33_u16..127).chain(161..173).chain(174..256) {
        alphabet.insert(char::from_u32(byte as u32).unwrap(), byte as u8);
        present[byte as usize] = true;
    }
    let mut extra = 256_u32;
    for (byte, &exists) in present.iter().enumerate() {
        if !exists {
            alphabet.insert(char::from_u32(extra).unwrap(), byte as u8);
            extra += 1;
        }
    }
    let mut decoded = vec![Vec::new(); 50257];
    for (token, id) in vocab {
        if id == 50256 {
            continue;
        }
        let destination = decoded.get_mut(id).ok_or("invalid GPT-2 vocabulary ID")?;
        for ch in token.chars() {
            destination.push(*alphabet.get(&ch).ok_or("invalid GPT-2 byte alphabet")?);
        }
    }
    Ok(decoded)
}
