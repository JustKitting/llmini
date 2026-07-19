use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    GPT2_N_EMBD, GPT2_N_LAYER, GPT2_TOKEN_ROWS, GPT2_VOCAB_SIZE, GPT2_XSA_ALPHA_STORAGE,
    Gpt2BackwardGrads, HiddenState, Logits,
};

use super::device_buffer::{block_array, zero};
use super::grad_block::{BlockGradBuffers, LayerNormGradBuffers};

pub struct BackwardBuffers {
    pub losses: DeviceBuffer<f32>,
    pub(super) d_lm_head_weight: DeviceBuffer<f32>,
    pub(super) dlogits: DeviceBuffer<f32>,
    pub(super) d_embedding_residual: DeviceBuffer<f32>,
    pub(super) d_xsa_alphas: DeviceBuffer<f32>,
    pub(super) blocks: [BlockGradBuffers; GPT2_N_LAYER],
    pub(super) final_norm: LayerNormGradBuffers,
}

pub struct BackwardParts<'a> {
    pub losses: &'a mut DeviceBuffer<f32>,
    pub d_lm_head_weight: &'a mut DeviceBuffer<f32>,
    pub grads: Gpt2BackwardGrads<'a>,
}

impl BackwardBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            losses: zero(stream, GPT2_TOKEN_ROWS)?,
            d_lm_head_weight: zero(stream, GPT2_VOCAB_SIZE * GPT2_N_EMBD)?,
            dlogits: zero(stream, Logits::LEN)?,
            d_embedding_residual: zero(stream, HiddenState::LEN)?,
            d_xsa_alphas: zero(stream, GPT2_XSA_ALPHA_STORAGE)?,
            blocks: block_array(|_| BlockGradBuffers::new(stream))?,
            final_norm: LayerNormGradBuffers::new(stream)?,
        })
    }

    pub fn parts(&mut self) -> BackwardParts<'_> {
        let blocks = self.blocks.as_mut_ptr();
        BackwardParts {
            losses: &mut self.losses,
            d_lm_head_weight: &mut self.d_lm_head_weight,
            grads: Gpt2BackwardGrads {
                dlogits: &mut self.dlogits,
                d_embedding_residual: &mut self.d_embedding_residual,
                d_xsa_alphas: &mut self.d_xsa_alphas,
                blocks: std::array::from_fn(|i| unsafe { (&mut *blocks.add(i)).grads() }),
                final_norm: self.final_norm.grads(),
            },
        }
    }
}
