use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, BlockAttentionBackwardScratch, GPT2_MLP, GPT2_N_EMBD, GPT2_QKV,
    GPT2_VOCAB_SIZE, Gpt2BackwardScratch, LinearScratch, MlpBackwardScratch,
};

pub struct BackwardScratchBuffers {
    final_head: LinearScratch,
    attention_c_proj: LinearScratch,
    attention_qkv: LinearScratch,
    pub attention_core: AttentionCoreScratchBuffers,
    mlp_down: LinearScratch,
    mlp_up: LinearScratch,
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
        })
    }

    pub fn scratch(&mut self) -> Gpt2BackwardScratch<'_> {
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
        }
    }
}
