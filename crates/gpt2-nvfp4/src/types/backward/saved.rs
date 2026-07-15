use cuda_core::DeviceBuffer;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use crate::GPT2_N_LAYER;

#[derive(Clone, Copy)]
pub struct Gpt2ForwardSaved<'a> {
    pub tokens: &'a DeviceBuffer<u32>,
    pub batch_size: u32,
    pub seq_len: u32,
    pub row_count: u32,
    pub blocks: [BlockForwardSaved<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormSaved<'a>,
    pub lm_head_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub logits: &'a DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub struct BlockForwardSaved<'a> {
    pub batch_size: u32,
    pub seq_len: u32,
    pub row_count: u32,
    pub ln_1: LayerNormSaved<'a>,
    pub qkv_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub qkv: &'a DeviceBuffer<u16>,
    pub attention_out: &'a DeviceBuffer<u16>,
    pub kda_v_new: Option<&'a DeviceBuffer<f32>>,
    pub attention_log_sum_exp: &'a DeviceBuffer<f32>,
    pub c_proj_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub ln_2: LayerNormSaved<'a>,
    pub mlp_up_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
    pub mlp_up: &'a DeviceBuffer<u16>,
    pub mlp_down_input_nvfp4: Nvfp4RowwiseDeviceTensor<'a>,
}

#[derive(Clone, Copy)]
pub struct LayerNormSaved<'a> {
    pub row_count: u32,
    pub residual: &'a DeviceBuffer<u16>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
}
