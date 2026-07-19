use cuda_core::CudaStream;
use gpt2_nvfp4::{
    AttentionProjectionTensors, Gpt2BlockWeights, MlpDownTensors, MlpProjectionTensors,
    MlpUpTensors,
};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::AppResult;

use super::{UploadedLayerNorm, UploadedLinear, UploadedNvfp4, tensor::upload_nvfp4};

pub struct UploadedBlock {
    pub ln_1: UploadedLayerNorm,
    pub attn_qkv: UploadedLinear,
    pub attn_qk_scale: UploadedNvfp4,
    pub attn_c_proj: UploadedLinear,
    pub ln_2: UploadedLayerNorm,
    pub mlp_up: UploadedLinear,
    pub mlp_down: UploadedLinear,
}

impl UploadedBlock {
    pub(in crate::upload) fn new(stream: &CudaStream, block: &Gpt2BlockWeights) -> AppResult<Self> {
        Ok(Self {
            ln_1: UploadedLayerNorm::from_layer_norm(stream, &block.ln_1)?,
            attn_qkv: UploadedLinear::from_linear(stream, &block.attn.c_attn)?,
            attn_qk_scale: upload_nvfp4(stream, &block.attn.qk_scale)?,
            attn_c_proj: UploadedLinear::from_linear(stream, &block.attn.c_proj)?,
            ln_2: UploadedLayerNorm::from_layer_norm(stream, &block.ln_2)?,
            mlp_up: UploadedLinear::from_linear(stream, &block.mlp.c_fc)?,
            mlp_down: UploadedLinear::from_linear(stream, &block.mlp.c_proj)?,
        })
    }

    pub fn attention_tensors<'a>(
        &'a self,
        xsa_alphas: Nvfp4DeviceTensor<'a>,
        xsa_alpha_offset: u32,
    ) -> AttentionProjectionTensors<'a> {
        AttentionProjectionTensors {
            qkv_weight: self.attn_qkv.weight.mma(),
            qkv_weight_device: self.attn_qkv.weight.device(),
            qkv_bias: self.attn_qkv.bias.device(),
            qk_scale: self.attn_qk_scale.device(),
            xsa_alphas,
            xsa_alpha_offset,
            c_proj_weight: self.attn_c_proj.weight.mma(),
            c_proj_weight_device: self.attn_c_proj.weight.device(),
            c_proj_bias: self.attn_c_proj.bias.device(),
        }
    }

    pub fn mlp_tensors(&self) -> MlpProjectionTensors<'_> {
        MlpProjectionTensors {
            up: MlpUpTensors {
                weight: self.mlp_up.weight.mma(),
                weight_device: self.mlp_up.weight.device(),
                bias: self.mlp_up.bias.device(),
            },
            down: MlpDownTensors {
                weight: self.mlp_down.weight.mma(),
                weight_device: self.mlp_down.weight.device(),
                bias: self.mlp_down.bias.device(),
            },
        }
    }
}
