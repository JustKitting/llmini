use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use cuda_core::CudaStream;

pub(crate) mod tensor;
mod topology;

use super::{format::CheckpointWriter, schema};
use crate::upload::UploadedModel;
use crate::{AppResult, fs_utils::ensure_parent};

pub fn save_uploaded_model(stream: &CudaStream, model: &UploadedModel, path: &Path) -> AppResult {
    validate_graph(model)?;
    ensure_parent(path)?;
    let file = File::create(path)?;
    write_uploaded_model_to(stream, model, file)
}

fn validate_graph(model: &UploadedModel) -> AppResult {
    if gpt2_nvfp4::GPT2_REQUIRES_ENGRAM_MHC
        && !(model.engram.is_some() && model.mhc.is_some() && model.latent_feedback.is_none())
    {
        return Err(
            "composite full-credit checkpoint requires the complete Engram/mHC graph".into(),
        );
    }
    if ((gpt2_nvfp4::GPT2_BOUNDED_SWIGLU && !gpt2_nvfp4::GPT2_BOUNDED_SWIGLU_ENGRAM_MHC)
        || (gpt2_nvfp4::GPT2_VALUE_RMS && !gpt2_nvfp4::GPT2_VALUE_RMS_ENGRAM_MHC)
        || (gpt2_nvfp4::GPT2_STATIONARY_ROPE && !gpt2_nvfp4::GPT2_STATIONARY_ROPE_ENGRAM_MHC))
        && (model.engram.is_some() || model.mhc.is_some() || model.latent_feedback.is_some())
    {
        return Err("this numerical/backward checkpoint encodes a separate native trial".into());
    }
    if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD
        && (model.engram.is_some() || model.mhc.is_some() || model.latent_feedback.is_some())
        && !(model.engram.is_some() && model.mhc.is_some() && model.latent_feedback.is_none())
    {
        return Err(
            "full-credit auxiliary checkpoint requires the complete Engram/mHC graph".into(),
        );
    }
    if model.mhc.is_some() && model.latent_feedback.is_some() {
        return Err("mHC and latent feedback are separate native checkpoint graphs".into());
    }
    if let Some(e) = &model.engram {
        let expected = if model.mhc.is_some() { 4 } else { 1 };
        if e.streams() != Some(expected) {
            return Err("Engram checkpoint stream graph is incompatible with its backbone".into());
        }
    }
    if model.engram.is_some() && model.latent_feedback.is_some() {
        return Err(
            "Engram and latent feedback are separate native architecture experiments".into(),
        );
    }
    Ok(())
}

pub(crate) fn write_uploaded_model_to(
    stream: &CudaStream,
    model: &UploadedModel,
    output: impl Write,
) -> AppResult {
    validate_graph(model)?;
    let mut writer = CheckpointWriter::new(BufWriter::new(output));
    writer.write_header(
        schema::tensor_count(model.blocks.len())
            + super::activation_quant::extra_tensors()
            + super::mlp::extra_tensors()
            + super::value_rms::extra_tensors()
            + super::kda_backward::extra_tensors()
            + super::kda_output_gate::extra_tensors()
            + super::stationary_rope::extra_tensors()
            + model.engram.as_ref().map_or(0, |w| w.tensor_count())
            + model.mhc.as_ref().map_or(0, |w| w.tensor_count())
            + if model.latent_feedback.is_some() {
                2
            } else {
                0
            },
    )?;
    topology::write_model(&mut writer, stream, model)?;
    super::activation_quant::write(&mut writer)?;
    super::mlp::write(&mut writer)?;
    super::value_rms::write(&mut writer)?;
    super::kda_backward::write(&mut writer)?;
    super::kda_output_gate::write(&mut writer)?;
    super::stationary_rope::write(&mut writer)?;
    if let Some(w) = &model.mhc {
        super::mhc::write(&mut writer, stream, w)?;
    }
    if let Some(weights) = &model.engram {
        super::engram::write(&mut writer, stream, weights)?;
    }
    if let Some(weights) = &model.latent_feedback {
        super::latent_feedback::write(&mut writer, stream, weights)?;
    }
    writer.finish()
}
