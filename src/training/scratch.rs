use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, BlockAttentionBackwardScratch, GPT2_MLP, GPT2_N_EMBD, GPT2_QKV,
    GPT2_VOCAB_SIZE, Gpt2BackwardScratch, HiddenState, LinearScratch, MlpBackwardScratch,
    QkvActivation,
};

use super::device_buffer::zero;

pub struct BackwardScratchBuffers {
    final_head: LinearScratch,
    attention_c_proj: LinearScratch,
    attention_qkv: LinearScratch,
    pub attention_core: AttentionCoreScratchBuffers,
    mlp_down: LinearScratch,
    mlp_up: LinearScratch,
    // Keep these disjoint from the forward residual and QKV workspaces. Matched
    // training diagnostics show that aliasing either workspace corrupts gradients.
    d_residual_after_attention: DeviceBuffer<f32>,
    d_qkv: DeviceBuffer<f32>,
}

impl BackwardScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            final_head: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_VOCAB_SIZE)?,
            attention_c_proj: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            attention_qkv: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            attention_core: AttentionCoreScratchBuffers::new(stream)?,
            mlp_down: LinearScratch::new(stream, GPT2_MLP, GPT2_N_EMBD)?,
            mlp_up: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_MLP)?,
            d_residual_after_attention: zero(stream, HiddenState::LEN)?,
            d_qkv: zero(stream, QkvActivation::LEN)?,
        })
    }

    pub fn scratch<'a>(
        &'a mut self,
        d_hidden: &'a mut DeviceBuffer<f32>,
        d_value_residual: &'a mut DeviceBuffer<f32>,
        d_mlp_up: &'a mut DeviceBuffer<f32>,
    ) -> Gpt2BackwardScratch<'a> {
        let down_linear = self.mlp_down.parts();
        let up_linear = self.mlp_up.parts();
        Gpt2BackwardScratch {
            final_head: self.final_head.final_head(),
            attention: BlockAttentionBackwardScratch {
                c_proj: self.attention_c_proj.c_proj(),
                core: self.attention_core.args(),
                qkv: self.attention_qkv.qkv(),
            },
            mlp: MlpBackwardScratch {
                down_linear,
                up_linear,
            },
            d_residual_after_attention: &mut self.d_residual_after_attention,
            d_hidden,
            d_qkv: &mut self.d_qkv,
            d_value_residual,
            d_mlp_up,
        }
    }
}
