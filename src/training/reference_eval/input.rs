use crate::{AppResult, reference_validation as reference};
use std::path::Path;

pub(super) struct Window {
    pub ids: Vec<u16>,
    pub lo: usize,
    pub hi: usize,
}

pub(super) fn load(path: &Path) -> AppResult<Vec<Window>> {
    if gpt2_nvfp4::GPT2_SEQ_LEN != reference::CONTEXT {
        return Err("reference evaluation requires the original 2048-token context".into());
    }
    let mut windows = Vec::new();
    reference::visit_documents(path, |ids| {
        let target_count = ids.len() - 1;
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
        Ok(())
    })?;
    if windows
        .iter()
        .map(|window| window.hi - window.lo)
        .sum::<usize>()
        != reference::TARGETS
    {
        return Err("reference target or byte accounting changed".into());
    }
    Ok(windows)
}
