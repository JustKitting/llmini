//! Exact positional geometry, checked before model upload or state mutation.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};
fn descriptor() -> [f32; 9] {
    // Version, head width, rotating width, base, inclusive endpoints, sign,
    // shifted first/last feature, native rotation-before-QK-normalization.
    [1., 64., 32., 1024., 1., -1., 32., 64., 1.]
}
pub(super) fn extra_tensors() -> u32 {
    gpt2_nvfp4::GPT2_STATIONARY_ROPE as u32
}
pub(super) fn write(writer: &mut CheckpointWriter<impl Write>) -> AppResult {
    if gpt2_nvfp4::GPT2_STATIONARY_ROPE {
        let values = descriptor();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_tensor("position.config", values.len(), 1., &bytes, &[])?;
    }
    Ok(())
}
pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    match tensors.remove("position.config") {
        None if !gpt2_nvfp4::GPT2_STATIONARY_ROPE => Ok(()),
        Some(t)
            if gpt2_nvfp4::GPT2_STATIONARY_ROPE
                && t.len == 9
                && t.bytes.len() == 36
                && t.global_scale == 1.
                && t.scales.is_empty()
                && t.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .eq(descriptor()) =>
        {
            Ok(())
        }
        _ => Err("checkpoint positional graph does not match this native executable".into()),
    }
}
