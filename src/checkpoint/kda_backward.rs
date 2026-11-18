//! Backward semantic identity; forward unchanged, fresh trial state required.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};
fn descriptor() -> [f32; 6] {
    [
        3., // Full state/decay, local query RHS and clamped-factor adjoints.
        1.,
        (gpt2_nvfp4::GPT2_N_EMBD / gpt2_nvfp4::GPT2_N_HEAD) as f32,
        gpt2_nvfp4::GPT2_SEQ_LEN as f32,
        (0..gpt2_nvfp4::GPT2_N_LAYER)
            .filter(|&i| !gpt2_nvfp4::uses_full_attention(i))
            .count() as f32,
        1.,
    ]
}
pub(super) fn extra_tensors() -> u32 {
    gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD as u32
}
pub(super) fn write(w: &mut CheckpointWriter<impl Write>) -> AppResult {
    if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        let values = descriptor();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        w.write_tensor("kda_backward.config", values.len(), 1., &bytes, &[])?;
    }
    Ok(())
}
pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    match tensors.remove("kda_backward.config") {
        None if !gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD => Ok(()),
        Some(t)
            if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
                && t.len == 6
                && t.bytes.len() == 24
                && t.scales.is_empty()
                && t.global_scale == 1.
                && t.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .eq(descriptor()) =>
        {
            Ok(())
        }
        _ => Err("checkpoint KDA backward semantics do not match this native executable".into()),
    }
}
