use std::collections::HashMap;

use cuda_core::CudaStream;
use gpt2_nvfp4::GPT2_N_LAYER;

use super::{super::format::CheckpointTensor, tensor::take_uploaded};
use crate::{
    AppResult,
    upload::{UploadedBlock, UploadedModel, UploadedNextLat, UploadedPair},
};

pub(super) fn load_model(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<UploadedModel> {
    Ok(UploadedModel {
        token_embedding: take_uploaded(stream, tensors, "token_embedding")?,
        blocks: load_blocks(stream, tensors)?,
        ln_f: load_pair(stream, tensors, "ln_f")?,
        next_latent: load_next_latent(stream, tensors)?,
    })
}

fn load_blocks(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<Vec<UploadedBlock>> {
    let mut blocks = Vec::with_capacity(GPT2_N_LAYER);
    for index in 0..GPT2_N_LAYER {
        blocks.push(UploadedBlock {
            ln_1: load_pair(stream, tensors, &format!("blocks.{index}.ln_1"))?,
            attn_qkv: load_pair(stream, tensors, &format!("blocks.{index}.attn_qkv"))?,
            attn_qk_scale: take_uploaded(
                stream,
                tensors,
                &format!("blocks.{index}.attn_qk_scale"),
            )?,
            attn_c_proj: load_pair(stream, tensors, &format!("blocks.{index}.attn_c_proj"))?,
            ln_2: load_pair(stream, tensors, &format!("blocks.{index}.ln_2"))?,
            mlp_up: load_pair(stream, tensors, &format!("blocks.{index}.mlp_up"))?,
            mlp_down: load_pair(stream, tensors, &format!("blocks.{index}.mlp_down"))?,
        });
    }
    Ok(blocks)
}

fn load_pair(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    prefix: &str,
) -> AppResult<UploadedPair> {
    Ok(UploadedPair::from_parts(
        take_uploaded(stream, tensors, &format!("{prefix}.weight"))?,
        take_uploaded(stream, tensors, &format!("{prefix}.bias"))?,
    ))
}

fn load_next_latent(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
) -> AppResult<UploadedNextLat> {
    Ok(UploadedNextLat {
        norm: load_pair(stream, tensors, "next_latent.norm")?,
        input_projection: load_pair(stream, tensors, "next_latent.input_projection")?,
        transition: load_pair(stream, tensors, "next_latent.transition")?,
        output_projection: load_pair(stream, tensors, "next_latent.output_projection")?,
    })
}
