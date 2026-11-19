use std::{collections::HashMap, io::Write};

use gpt2_nvfp4::GPT2_KDA_RATIONAL_OUTPUT_GATE;

use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;

const NAME: &str = "kda_output_gate.config";
const DESCRIPTOR: [u8; 4] = 1.0_f32.to_le_bytes();

pub(super) fn extra_tensors() -> u32 {
    u32::from(GPT2_KDA_RATIONAL_OUTPUT_GATE)
}

pub(super) fn write(writer: &mut CheckpointWriter<impl Write>) -> AppResult {
    if GPT2_KDA_RATIONAL_OUTPUT_GATE {
        writer.write_tensor(NAME, 1, 1.0, &DESCRIPTOR, &[])?;
    }
    Ok(())
}

pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    match tensors.remove(NAME) {
        None if !GPT2_KDA_RATIONAL_OUTPUT_GATE => Ok(()),
        Some(tensor)
            if GPT2_KDA_RATIONAL_OUTPUT_GATE
                && tensor.len == 1
                && tensor.global_scale == 1.0
                && tensor.scales.is_empty()
                && tensor.bytes == DESCRIPTOR =>
        {
            Ok(())
        }
        _ => Err("checkpoint KDA output gate does not match this native executable".into()),
    }
}
