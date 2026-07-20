use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    BlockBackwardGrads, GPT2_CANON_WEIGHT_COUNT, GPT2_MLP, GPT2_N_EMBD, GPT2_QK_SCALE_STORAGE,
    GPT2_QKV,
};

use super::LayerNormGradBuffers;
use crate::training::device_buffer::zero;

pub struct BlockGradBuffers {
    pub(in crate::training) ln_1: LayerNormGradBuffers,
    pub(in crate::training) d_canon_a_weight: DeviceBuffer<f32>,
    pub(in crate::training) ln_2: LayerNormGradBuffers,
    pub(in crate::training) d_canon_c_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_attn_qkv_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_attn_qkv_bias: DeviceBuffer<f32>,
    pub(in crate::training) d_attn_qk_scale: DeviceBuffer<f32>,
    pub(in crate::training) d_attn_c_proj_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_attn_c_proj_bias: DeviceBuffer<f32>,
    pub(in crate::training) d_mlp_c_fc_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_mlp_c_fc_bias: DeviceBuffer<f32>,
    pub(in crate::training) d_mlp_c_proj_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_mlp_c_proj_bias: DeviceBuffer<f32>,
}

impl BlockGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormGradBuffers::new(stream)?,
            d_canon_a_weight: zero(stream, GPT2_CANON_WEIGHT_COUNT)?,
            ln_2: LayerNormGradBuffers::new(stream)?,
            d_canon_c_weight: zero(stream, GPT2_CANON_WEIGHT_COUNT)?,
            d_attn_qkv_weight: zero(stream, GPT2_N_EMBD * GPT2_QKV)?,
            d_attn_qkv_bias: zero(stream, GPT2_QKV)?,
            d_attn_qk_scale: zero(stream, GPT2_QK_SCALE_STORAGE)?,
            d_attn_c_proj_weight: zero(stream, GPT2_N_EMBD * GPT2_N_EMBD)?,
            d_attn_c_proj_bias: zero(stream, GPT2_N_EMBD)?,
            d_mlp_c_fc_weight: zero(stream, GPT2_N_EMBD * GPT2_MLP)?,
            d_mlp_c_fc_bias: zero(stream, GPT2_MLP)?,
            d_mlp_c_proj_weight: zero(stream, GPT2_MLP * GPT2_N_EMBD)?,
            d_mlp_c_proj_bias: zero(stream, GPT2_N_EMBD)?,
        })
    }

    pub fn grads(&mut self) -> BlockBackwardGrads<'_> {
        BlockBackwardGrads {
            ln_1: self.ln_1.grads(),
            d_canon_a_weight: &mut self.d_canon_a_weight,
            ln_2: self.ln_2.grads(),
            d_canon_c_weight: &mut self.d_canon_c_weight,
            d_attn_qkv_weight: &mut self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut self.d_attn_qkv_bias,
            d_attn_qk_scale: &mut self.d_attn_qk_scale,
            d_attn_c_proj_weight: &mut self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut self.d_mlp_c_proj_bias,
        }
    }
}
