use crate::{AppResult, reference_validation as reference};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

pub(super) struct Window {
    pub ids: Vec<u16>,
    pub lo: usize,
    pub hi: usize,
}

pub(super) fn load(path: &Path) -> AppResult<Vec<Window>> {
    reference::require_tokenizer()?;
    reference::require_hash(path, reference::DATA_SHA)?;
    if gpt2_nvfp4::GPT2_SEQ_LEN != reference::CONTEXT {
        return Err("reference evaluation requires the original 2048-token context".into());
    }
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = vec![0_u8; reference::MAGIC.len()];
    input.read_exact(&mut magic)?;
    if magic != reference::MAGIC {
        return Err("invalid reference document magic".into());
    }
    let documents = read_u32(&mut input)?;
    if documents != reference::DOCUMENTS || read_u32(&mut input)? != reference::CONTEXT {
        return Err("reference document shape changed".into());
    }
    let mut windows = Vec::new();
    let mut raw_bytes = 0;
    let mut targets = 0;
    for document in 0..documents {
        if read_u32(&mut input)? != document {
            return Err("reference document order changed".into());
        }
        raw_bytes += read_u32(&mut input)?;
        let closed = read_u32(&mut input)?;
        if closed != usize::from(document + 1 < documents) {
            return Err("reference document termination changed".into());
        }
        let count = read_u32(&mut input)?;
        if count < 2 || count > 10_000_000 {
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
        let target_count = count - 1;
        targets += target_count;
        let mut scored = 0;
        while scored < target_count {
            let count = if scored == 0 {
                target_count.min(reference::CONTEXT)
            } else {
                (target_count - scored).min(reference::STRIDE)
            };
            let start = (scored + count).saturating_sub(reference::CONTEXT);
            let length = (target_count - start).min(reference::CONTEXT);
            let mut padded = vec![0_u16; reference::CONTEXT + 1];
            padded[..length + 1].copy_from_slice(&ids[start..start + length + 1]);
            let lo = scored - start;
            let hi = lo + count;
            if hi > length {
                return Err("reference window exceeds available targets".into());
            }
            windows.push(Window {
                ids: padded,
                lo,
                hi,
            });
            scored += count;
        }
    }
    let mut extra = [0_u8; 1];
    if input.read(&mut extra)? != 0 {
        return Err("unexpected reference trailing bytes".into());
    }
    if raw_bytes != reference::RAW_BYTES
        || targets != reference::TARGETS
        || windows
            .iter()
            .map(|window| window.hi - window.lo)
            .sum::<usize>()
            != targets
    {
        return Err("reference target or byte accounting changed".into());
    }
    Ok(windows)
}

fn read_u32(input: &mut impl Read) -> AppResult<usize> {
    let mut bytes = [0_u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes) as usize)
}
