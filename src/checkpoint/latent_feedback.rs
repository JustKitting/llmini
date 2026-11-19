//! Same-file physical NVFP4 feedback map and exact algorithm descriptor.
use super::{
    format::{CheckpointTensor, CheckpointWriter},
    load::tensor::take_uploaded,
    save::tensor::write as write_nv,
};
use crate::{
    AppResult,
    training::latent_feedback::{EPSILON, EVAL_PASSES, ONE_PASS_PROBABILITY, Weights},
};
use cuda_core::CudaStream;
use gpt2_nvfp4::GPT2_N_EMBD;
use std::{collections::HashMap, io::Write};

pub(super) const TENSOR_COUNT: u32 = 2;

fn config() -> Vec<f32> {
    vec![
        1.0,
        GPT2_N_EMBD as f32,
        EPSILON,
        ONE_PASS_PROBABILITY,
        EVAL_PASSES as f32,
        0.02,
    ]
}
pub(super) fn write(
    writer: &mut CheckpointWriter<impl Write>,
    s: &CudaStream,
    w: &Weights,
) -> AppResult {
    write_nv(writer, s, "latent_feedback.fusion.weight", &w.weight)?;
    let values = config();
    let bytes = values
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_tensor("latent_feedback.config", values.len(), 1.0, &bytes, &[])
}
pub(super) fn load(
    s: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<Weights> {
    let c = tensors
        .remove("latent_feedback.config")
        .ok_or("missing feedback architecture descriptor")?;
    let expected = config();
    let bytes = expected
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    if c.len != expected.len() || c.bytes != bytes || !c.scales.is_empty() {
        return Err("feedback architecture descriptor mismatch".into());
    }
    let weight = take_uploaded(s, tensors, "latent_feedback.fusion.weight")?;
    if weight.len != GPT2_N_EMBD * GPT2_N_EMBD
        || weight.scale_rows != GPT2_N_EMBD as u32
        || weight.scale_k != GPT2_N_EMBD as u32
    {
        return Err("feedback matrix shape/CSF layout mismatch".into());
    }
    Ok(Weights { weight })
}
