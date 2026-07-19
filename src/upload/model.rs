use cuda_core::CudaStream;
use gpt2_nvfp4::{Gpt2BackwardWeights, Gpt2Weights};

use crate::AppResult;

use super::{
    UploadedBlock, UploadedLayerNorm, UploadedNextLat, UploadedNvfp4, tensor::upload_nvfp4,
};

pub struct UploadedModel {
    pub token_embedding: UploadedNvfp4,
    pub xsa_alphas: UploadedNvfp4,
    pub blocks: Vec<UploadedBlock>,
    pub ln_f: UploadedLayerNorm,
    pub next_latent: UploadedNextLat,
}

impl UploadedModel {
    pub fn new(stream: &CudaStream, weights: &Gpt2Weights) -> AppResult<Self> {
        Ok(Self {
            token_embedding: upload_nvfp4(stream, &weights.embeddings.wte)?,
            xsa_alphas: upload_nvfp4(stream, &weights.xsa_alphas)?,
            blocks: weights
                .h
                .iter()
                .map(|block| UploadedBlock::new(stream, block))
                .collect::<AppResult<_>>()?,
            ln_f: UploadedLayerNorm::from_layer_norm(stream, &weights.ln_f)?,
            next_latent: UploadedNextLat::new(stream, &weights.next_latent)?,
        })
    }

    pub fn backward_weights(&self) -> Gpt2BackwardWeights<'_> {
        Gpt2BackwardWeights {
            lm_head_weight: self.token_embedding.device(),
            ln_f: self.ln_f.tensors(),
            block_ln_1: std::array::from_fn(|i| self.blocks[i].ln_1.tensors()),
            block_ln_2: std::array::from_fn(|i| self.blocks[i].ln_2.tensors()),
            attention: std::array::from_fn(|i| {
                self.blocks[i].attention_tensors(
                    self.xsa_alphas.device(),
                    (i * gpt2_nvfp4::GPT2_N_HEAD) as u32,
                )
            }),
            mlp: std::array::from_fn(|i| self.blocks[i].mlp_tensors()),
        }
    }
}
