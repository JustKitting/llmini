use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionLogSumExp, BlockForwardSaved, BlockForwardTape, GPT2_BATCH_SIZE, GPT2_MLP_ROUTE_MASKS,
    GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN, HiddenState, MlpActivation, QkvActivation,
    attention_headwise_gate_enabled, uses_full_attention,
};

use super::device_buffer::zero;
use super::tape_leaf::{LayerNormTapeBuffers, RowwiseTapeBuffers};

pub struct BlockTapeBuffers {
    ln_1: LayerNormTapeBuffers,
    qkv_input: RowwiseTapeBuffers,
    qkv: DeviceBuffer<u16>,
    attention_out: DeviceBuffer<u16>,
    headwise_gate_input: Option<DeviceBuffer<u16>>,
    attention_probs: Option<DeviceBuffer<u16>>,
    kda_v_new: Option<DeviceBuffer<f32>>,
    kda_akk_inv: Option<DeviceBuffer<f32>>,
    kda_w: Option<DeviceBuffer<f32>>,
    kda_aqk: Option<DeviceBuffer<f32>>,
    attention_log_sum_exp: DeviceBuffer<f32>,
    c_proj_input: RowwiseTapeBuffers,
    ln_2: LayerNormTapeBuffers,
    mlp_up_input: RowwiseTapeBuffers,
    mlp_up: DeviceBuffer<u16>,
    mlp_down_input: RowwiseTapeBuffers,
    mlp_route_masks: DeviceBuffer<u64>,
}

impl BlockTapeBuffers {
    pub fn new(stream: &CudaStream, block_index: usize) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormTapeBuffers::new(stream)?,
            qkv_input: RowwiseTapeBuffers::gpt2_rows(stream, HiddenState::LEN)?,
            qkv: zero(stream, QkvActivation::LEN)?,
            attention_out: zero(stream, HiddenState::LEN)?,
            headwise_gate_input: if attention_headwise_gate_enabled()
                && !uses_full_attention(block_index)
            {
                Some(zero(stream, HiddenState::LEN)?)
            } else {
                None
            },
            attention_probs: if uses_full_attention(block_index) && block_index != GPT2_N_LAYER - 1
            {
                Some(zero(
                    stream,
                    GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN,
                )?)
            } else {
                None
            },
            kda_v_new: if uses_full_attention(block_index) {
                None
            } else {
                Some(zero(stream, HiddenState::LEN)?)
            },
            kda_akk_inv: if uses_full_attention(block_index) {
                None
            } else {
                Some(zero(stream, HiddenState::LEN)?)
            },
            kda_w: if uses_full_attention(block_index) {
                None
            } else {
                Some(zero(stream, HiddenState::LEN)?)
            },
            kda_aqk: if uses_full_attention(block_index) {
                None
            } else {
                Some(zero(stream, HiddenState::LEN)?)
            },
            attention_log_sum_exp: zero(stream, AttentionLogSumExp::LEN)?,
            c_proj_input: RowwiseTapeBuffers::gpt2_rows(stream, HiddenState::LEN)?,
            ln_2: LayerNormTapeBuffers::new(stream)?,
            mlp_up_input: RowwiseTapeBuffers::gpt2_rows(stream, HiddenState::LEN)?,
            mlp_up: zero(stream, MlpActivation::LEN)?,
            mlp_down_input: RowwiseTapeBuffers::gpt2_rows(stream, MlpActivation::LEN)?,
            mlp_route_masks: zero(stream, GPT2_MLP_ROUTE_MASKS)?,
        })
    }

    pub fn tape(&mut self) -> BlockForwardTape<'_> {
        BlockForwardTape {
            ln_1: self.ln_1.tape(),
            qkv_input_nvfp4: self.qkv_input.tape(),
            qkv: &mut self.qkv,
            attention_out: &mut self.attention_out,
            headwise_gate_input: self.headwise_gate_input.as_mut(),
            attention_probs: self.attention_probs.as_mut(),
            kda_v_new: self.kda_v_new.as_mut(),
            kda_akk_inv: self.kda_akk_inv.as_mut(),
            kda_w: self.kda_w.as_mut(),
            kda_aqk: self.kda_aqk.as_mut(),
            attention_log_sum_exp: &mut self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input.tape(),
            ln_2: self.ln_2.tape(),
            mlp_up_input_nvfp4: self.mlp_up_input.tape(),
            mlp_up: &mut self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input.tape(),
            mlp_route_masks: &mut self.mlp_route_masks,
        }
    }

    pub fn saved(&self, batch_size: u32, seq_len: u32, row_count: u32) -> BlockForwardSaved<'_> {
        BlockForwardSaved {
            batch_size,
            seq_len,
            row_count,
            ln_1: self.ln_1.saved(row_count),
            qkv_input_nvfp4: self.qkv_input.saved(),
            qkv: &self.qkv,
            attention_out: &self.attention_out,
            headwise_gate_input: self.headwise_gate_input.as_ref(),
            attention_probs: self.attention_probs.as_ref(),
            kda_v_new: self.kda_v_new.as_ref(),
            kda_akk_inv: self.kda_akk_inv.as_ref(),
            kda_w: self.kda_w.as_ref(),
            kda_aqk: self.kda_aqk.as_ref(),
            attention_log_sum_exp: &self.attention_log_sum_exp,
            c_proj_input_nvfp4: self.c_proj_input.saved(),
            ln_2: self.ln_2.saved(row_count),
            mlp_up_input_nvfp4: self.mlp_up_input.saved(),
            mlp_up: &self.mlp_up,
            mlp_down_input_nvfp4: self.mlp_down_input.saved(),
            mlp_route_masks: &self.mlp_route_masks,
        }
    }

    pub fn qkv(&self) -> &DeviceBuffer<u16> {
        &self.qkv
    }
}
