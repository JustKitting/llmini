#![allow(dead_code)]

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    BlockForwardSaved, GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, LayerNormSaved,
};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

pub struct SavedBlockParts<'a> {
    pub rowwise: Nvfp4RowwiseDeviceTensor<'a>,
    pub residual: &'a DeviceBuffer<u16>,
    pub mean: &'a DeviceBuffer<f32>,
    pub inv_std: &'a DeviceBuffer<f32>,
    pub qkv: &'a DeviceBuffer<u16>,
    pub attention_out: &'a DeviceBuffer<u16>,
    pub attention_log_sum_exp: &'a DeviceBuffer<f32>,
    pub mlp_up: &'a DeviceBuffer<u16>,
}

pub fn saved_block(parts: SavedBlockParts<'_>) -> BlockForwardSaved<'_> {
    let ln = LayerNormSaved {
        row_count: GPT2_TOKEN_ROWS as u32,
        residual: parts.residual,
        mean: parts.mean,
        inv_std: parts.inv_std,
    };
    BlockForwardSaved {
        batch_size: GPT2_BATCH_SIZE as u32,
        seq_len: GPT2_SEQ_LEN as u32,
        row_count: GPT2_TOKEN_ROWS as u32,
        ln_1: ln,
        qkv_input_nvfp4: parts.rowwise,
        qkv: parts.qkv,
        attention_out: parts.attention_out,
        attention_probs: None,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        attention_log_sum_exp: parts.attention_log_sum_exp,
        c_proj_input_nvfp4: parts.rowwise,
        ln_2: ln,
        mlp_up_input_nvfp4: parts.rowwise,
        mlp_up: parts.mlp_up,
        mlp_down_input_nvfp4: parts.rowwise,
    }
}
