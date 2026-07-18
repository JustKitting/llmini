use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    BlockBackwardGrads, GPT2_MLP, GPT2_N_EMBD, GPT2_QK_SCALE_STORAGE, GPT2_QKV, HiddenState,
    LayerNormGrads, QkvActivation,
};

use crate::data;

pub struct GradBuffers {
    pub d_residual_in: DeviceBuffer<f32>,
    pub d_hidden: DeviceBuffer<f32>,
    pub d_qkv: DeviceBuffer<f32>,
    pub d_attn_qkv_weight: DeviceBuffer<f32>,
    pub d_attn_qkv_bias: DeviceBuffer<f32>,
    pub d_attn_qk_scale: DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: DeviceBuffer<f32>,
    pub d_attn_c_proj_bias: DeviceBuffer<f32>,
    d_residual_after_attention: DeviceBuffer<f32>,
    ln1: LayerNormGradBuffers,
    ln2: LayerNormGradBuffers,
    d_mlp_c_fc_weight: DeviceBuffer<f32>,
    d_mlp_c_fc_bias: DeviceBuffer<f32>,
    d_mlp_c_proj_weight: DeviceBuffer<f32>,
    d_mlp_c_proj_bias: DeviceBuffer<f32>,
}

impl GradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_residual_in: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_hidden: DeviceBuffer::zeroed(stream, HiddenState::LEN)?,
            d_qkv: DeviceBuffer::zeroed(stream, QkvActivation::LEN)?,
            d_attn_qkv_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_QKV)?,
            d_attn_qkv_bias: DeviceBuffer::zeroed(stream, GPT2_QKV)?,
            d_attn_qk_scale: DeviceBuffer::zeroed(stream, GPT2_QK_SCALE_STORAGE)?,
            d_attn_c_proj_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_N_EMBD)?,
            d_attn_c_proj_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
            d_residual_after_attention: DeviceBuffer::from_host(stream, &data::hidden_values())?,
            ln1: LayerNormGradBuffers::new(stream)?,
            ln2: LayerNormGradBuffers::new(stream)?,
            d_mlp_c_fc_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD * GPT2_MLP)?,
            d_mlp_c_fc_bias: DeviceBuffer::zeroed(stream, GPT2_MLP)?,
            d_mlp_c_proj_weight: DeviceBuffer::zeroed(stream, GPT2_MLP * GPT2_N_EMBD)?,
            d_mlp_c_proj_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
        })
    }

    pub fn block(
        &mut self,
    ) -> (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        BlockBackwardGrads<'_>,
    ) {
        (
            &mut self.d_residual_after_attention,
            &mut self.d_residual_in,
            &mut self.d_hidden,
            &mut self.d_qkv,
            BlockBackwardGrads {
                ln_1: self.ln1.grads(),
                ln_2: self.ln2.grads(),
                d_attn_qkv_weight: &mut self.d_attn_qkv_weight,
                d_attn_qkv_bias: &mut self.d_attn_qkv_bias,
                d_attn_qk_scale: &mut self.d_attn_qk_scale,
                d_attn_c_proj_weight: &mut self.d_attn_c_proj_weight,
                d_attn_c_proj_bias: &mut self.d_attn_c_proj_bias,
                d_mlp_c_fc_weight: &mut self.d_mlp_c_fc_weight,
                d_mlp_c_fc_bias: &mut self.d_mlp_c_fc_bias,
                d_mlp_c_proj_weight: &mut self.d_mlp_c_proj_weight,
                d_mlp_c_proj_bias: &mut self.d_mlp_c_proj_bias,
            },
        )
    }
}

struct LayerNormGradBuffers {
    d_weight: DeviceBuffer<f32>,
    d_bias: DeviceBuffer<f32>,
}

impl LayerNormGradBuffers {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_weight: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
            d_bias: DeviceBuffer::zeroed(stream, GPT2_N_EMBD)?,
        })
    }

    fn grads(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_weight: &mut self.d_weight,
            d_bias: &mut self.d_bias,
        }
    }
}
