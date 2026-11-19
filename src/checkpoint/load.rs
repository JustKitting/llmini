use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use cuda_core::CudaStream;
use gpt2_nvfp4::GPT2_N_LAYER;

mod reuse;
pub(crate) mod tensor;
mod topology;

pub(crate) use reuse::prepare_reload;

use super::{format::CheckpointReader, schema};
use crate::AppResult;
use crate::upload::UploadedModel;

fn read_tensors(input: impl Read) -> AppResult<HashMap<String, super::format::CheckpointTensor>> {
    let mut reader = CheckpointReader::new(BufReader::new(input));
    let tensor_count = reader.read_header()?;
    let expected_tensor_count = schema::tensor_count(GPT2_N_LAYER);
    if !schema::supported_tensor_count(GPT2_N_LAYER, tensor_count) {
        return Err(format!(
            "checkpoint has {tensor_count} tensors; expected {expected_tensor_count}"
        )
        .into());
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let tensor = reader.read_tensor()?;
        if tensors.insert(tensor.name.clone(), tensor).is_some() {
            return Err("checkpoint contains duplicate tensor name".into());
        }
    }

    super::activation_quant::take_and_validate(&mut tensors)?;
    super::mlp::take_and_validate(&mut tensors)?;
    super::value_rms::take_and_validate(&mut tensors)?;
    super::kda_backward::take_and_validate(&mut tensors)?;
    super::kda_output_gate::take_and_validate(&mut tensors)?;
    super::stationary_rope::take_and_validate(&mut tensors)?;
    if gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC
        && !(tensors.contains_key("mhc.config")
            && tensors.contains_key("engram.layers")
            && !tensors.contains_key("latent_feedback.config"))
    {
        return Err(
            "composite full-credit checkpoint requires the complete Engram/mHC graph".into(),
        );
    }
    if ((gpt2_nvfp4::GPT2_VALUE_RMS && !gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC)
        || (gpt2_nvfp4::GPT2_STATIONARY_ROPE && !gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC))
        && [
            "mhc.config",
            "engram.config",
            "engram.layers",
            "latent_feedback.config",
        ]
        .iter()
        .any(|key| tensors.contains_key(*key))
    {
        return Err(
            "this numerical/backward checkpoint must encode its separate native trial".into(),
        );
    }
    if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
        && !schema::full_credit_supports_graph(
            tensors.contains_key("engram.config") || tensors.contains_key("engram.layers"),
            tensors.contains_key("mhc.config"),
            tensors.contains_key("latent_feedback.config"),
        )
    {
        return Err(
            "full-credit auxiliary checkpoint requires the complete Engram/mHC graph".into(),
        );
    }
    Ok(tensors)
}

pub fn load_uploaded_model(stream: &CudaStream, path: &Path) -> AppResult<UploadedModel> {
    load_uploaded_model_from_reader(stream, File::open(path)?)
}

pub(crate) fn load_uploaded_model_from_reader(
    stream: &CudaStream,
    input: impl Read,
) -> AppResult<UploadedModel> {
    let mut tensors = read_tensors(input)?;
    let mut model = topology::load_model(stream, &mut tensors)?;
    if tensors.contains_key("mhc.config") {
        model.mhc = Some(super::mhc::load(stream, &mut tensors)?);
    }
    if tensors.contains_key("engram.config") || tensors.contains_key("engram.layers") {
        model.engram = Some(super::engram::load(stream, &mut tensors)?);
    }
    if tensors.contains_key("latent_feedback.config") {
        model.latent_feedback = Some(super::latent_feedback::load(stream, &mut tensors)?);
    }
    if !tensors.is_empty() {
        return Err("checkpoint has unrecognized tensors".into());
    }
    if model.mhc.is_some() && model.latent_feedback.is_some() {
        return Err("unsupported mixed mHC checkpoint graph".into());
    }
    if let Some(e) = &model.engram {
        let expected = if model.mhc.is_some() { 4 } else { 1 };
        if e.streams() != Some(expected) {
            return Err("Engram checkpoint stream graph is incompatible with its backbone".into());
        }
    }
    Ok(model)
}
