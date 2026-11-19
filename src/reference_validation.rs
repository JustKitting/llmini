//! Identities and byte accounting for the reserved NanoGPT reference text.
use crate::AppResult;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

pub const MAGIC: &[u8] = b"NVFP4_NANOGPT_BPB_V1\n";
pub const SOURCE_TARGETS: usize = 10_485_760;
pub const RAW_BYTES: usize = 46_515_256;
pub const DOCUMENTS: usize = 14_983;
pub const TARGETS: usize = 11_529_614;
pub const CONTEXT: usize = 2048;
pub const STRIDE: usize = 512;
pub const SOURCE_SHA: &str = "5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640";
pub const VOCAB_SHA: &str = "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783";
pub const DATA_SHA: &str = "e8db0ff1965c1c02c0a46e95ae0d5ea1b3988adcf6580d21b0042d0c95d59de5";
pub const RAW_SHA: &str = "1eb8ad192e96f36f3d5c0cdf585c015d9674f80bd3341b0bdfe60069c115a406";
pub const TOKENIZER_SHA: &str = "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720";

pub fn sha256(path: &Path) -> AppResult<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn require_hash(path: &Path, expected: &str) -> AppResult {
    let actual = sha256(path)?;
    if actual != expected {
        return Err(format!("{}: SHA256 {actual}, expected {expected}", path.display()).into());
    }
    Ok(())
}

pub fn require_tokenizer() -> AppResult {
    if llama2_tokenizer::TOKENIZER_NAME != "mistral_v01"
        || llama2_tokenizer::MODEL_VOCAB_SIZE != 32000
    {
        return Err("reference documents require the unchanged Mistral v0.1 tokenizer".into());
    }
    let tokenizer = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates/llama2-tokenizer/assets/tokenizers/mistral_v01/tokenizer.json");
    require_hash(&tokenizer, TOKENIZER_SHA)
}

pub fn visit_documents(path: &Path, mut visit: impl FnMut(&[u16]) -> AppResult) -> AppResult {
    require_tokenizer()?;
    require_hash(path, DATA_SHA)?;
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = vec![0_u8; MAGIC.len()];
    input.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err("invalid reference document magic".into());
    }
    if read_u32(&mut input)? != DOCUMENTS || read_u32(&mut input)? != CONTEXT {
        return Err("reference document shape changed".into());
    }
    let mut raw_bytes = 0;
    let mut targets = 0;
    for document in 0..DOCUMENTS {
        if read_u32(&mut input)? != document {
            return Err("reference document order changed".into());
        }
        raw_bytes += read_u32(&mut input)?;
        let closed = read_u32(&mut input)?;
        if closed != usize::from(document + 1 < DOCUMENTS) {
            return Err("reference document termination changed".into());
        }
        let count = read_u32(&mut input)?;
        if !(2..=10_000_000).contains(&count) {
            return Err("invalid document token count".into());
        }
        let mut bytes = vec![0_u8; 2 * count];
        input.read_exact(&mut bytes)?;
        let ids = bytes
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes([x[0], x[1]]))
            .collect::<Vec<_>>();
        if ids[0] != 1 || (closed == 1 && ids[count - 1] != 2) || ids.iter().any(|&v| v >= 32000) {
            return Err("reference document token policy changed".into());
        }
        targets += ids.len() - 1;
        visit(&ids)?;
    }
    if input.read(&mut [0_u8; 1])? != 0 {
        return Err("unexpected reference trailing bytes".into());
    }
    if raw_bytes != RAW_BYTES || targets != TARGETS {
        return Err("reference target or byte accounting changed".into());
    }
    Ok(())
}

fn read_u32(input: &mut impl Read) -> AppResult<usize> {
    let mut bytes = [0_u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}
