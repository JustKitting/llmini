use cuda_core::DeviceBuffer;

use super::LayerNormGrads;

pub struct BlockBackwardGrads<'a> {
    pub ln_1: LayerNormGrads<'a>,
    pub ln_2: LayerNormGrads<'a>,
    pub d_attn_qkv_weight: &'a mut DeviceBuffer<f32>,
    pub d_attn_qkv_bias: &'a mut DeviceBuffer<f32>,
    pub d_attn_qk_scale: &'a mut DeviceBuffer<f32>,
    pub d_attn_c_proj_weight: &'a mut DeviceBuffer<f32>,
    pub d_attn_c_proj_bias: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_fc_weight: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_fc_bias: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_proj_weight: &'a mut DeviceBuffer<f32>,
    pub d_mlp_c_proj_bias: &'a mut DeviceBuffer<f32>,
}

macro_rules! reborrow_block_grads {
    ($self:expr) => {
        BlockBackwardGrads {
            ln_1: $self.ln_1.reborrow(),
            ln_2: $self.ln_2.reborrow(),
            d_attn_qkv_weight: &mut *$self.d_attn_qkv_weight,
            d_attn_qkv_bias: &mut *$self.d_attn_qkv_bias,
            d_attn_qk_scale: &mut *$self.d_attn_qk_scale,
            d_attn_c_proj_weight: &mut *$self.d_attn_c_proj_weight,
            d_attn_c_proj_bias: &mut *$self.d_attn_c_proj_bias,
            d_mlp_c_fc_weight: &mut *$self.d_mlp_c_fc_weight,
            d_mlp_c_fc_bias: &mut *$self.d_mlp_c_fc_bias,
            d_mlp_c_proj_weight: &mut *$self.d_mlp_c_proj_weight,
            d_mlp_c_proj_bias: &mut *$self.d_mlp_c_proj_bias,
        }
    };
}

impl<'a> BlockBackwardGrads<'a> {
    pub fn reborrow(&mut self) -> BlockBackwardGrads<'_> {
        reborrow_block_grads!(self)
    }
}
