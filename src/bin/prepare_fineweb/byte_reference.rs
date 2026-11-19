use crate::{AppResult, reference_validation as reference};
use llama2_tokenizer::TrainingTokenizer;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

pub(super) const MAGIC: &[u8] = b"NVFP4_REFERENCE_BYTES_V1\n";

pub(super) fn write_document(output: &mut impl Write, raw: &[u8], closed: bool) -> AppResult {
    output.write_all(&u32::try_from(raw.len())?.to_le_bytes())?;
    output.write_all(&[u8::from(closed)])?;
    output.write_all(raw)?;
    Ok(())
}

/// Export the exact text from the verified tokenized reference. A record has
/// byte length, termination flag, then its original bytes. BOS and EOS are
/// represented structurally, independent of the chosen byte vocabulary.
pub fn prepare(source: &Path, output: &Path) -> AppResult {
    if output.exists() {
        return Err(format!("{} already exists", output.display()).into());
    }
    let tokenizer = TrainingTokenizer::from_default_assets()?;
    let mut partial_name = output.as_os_str().to_owned();
    partial_name.push(".partial");
    let partial = Path::new(&partial_name);
    let writer = BufWriter::new(File::create_new(partial)?);
    let generated = export(source, writer, &tokenizer);
    if let Err(error) = generated {
        let _ = fs::remove_file(partial);
        return Err(error);
    }
    fs::rename(partial, output)?;
    println!("byte_reference_documents={}", output.display());
    println!(
        "documents={} raw_bytes={} raw_sha256={}",
        reference::DOCUMENTS,
        reference::RAW_BYTES,
        reference::RAW_SHA
    );
    Ok(())
}

fn export(source: &Path, mut output: impl Write, tokenizer: &TrainingTokenizer) -> AppResult {
    output.write_all(MAGIC)?;
    output.write_all(&u32::try_from(reference::DOCUMENTS)?.to_le_bytes())?;
    let mut index = 0;
    let mut raw_hash = Sha256::new();
    let mut raw_bytes = 0;
    reference::visit_documents(source, |ids| {
        let closed = index + 1 < reference::DOCUMENTS;
        let text_ids: Vec<_> = ids[1..ids.len() - usize::from(closed)]
            .iter()
            .map(|&id| u32::from(id))
            .collect();
        let text = tokenizer.decode(&text_ids)?;
        let raw = text.as_bytes();
        write_document(&mut output, raw, closed)?;
        raw_hash.update(raw);
        raw_bytes += raw.len();
        index += 1;
        Ok(())
    })?;
    if index != reference::DOCUMENTS
        || raw_bytes != reference::RAW_BYTES
        || format!("{:x}", raw_hash.finalize()) != reference::RAW_SHA
    {
        return Err("byte export differs from the reserved raw FineWeb reference".into());
    }
    output.flush()?;
    Ok(())
}
