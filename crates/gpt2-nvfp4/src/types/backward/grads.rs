use cuda_core::DeviceBuffer;

use crate::GPT2_N_LAYER;

#[path = "grads/block.rs"]
mod block;
#[path = "grads/layer_norm.rs"]
mod layer_norm;

pub use block::BlockBackwardGrads;
pub use layer_norm::LayerNormGrads;

pub struct Gpt2BackwardGrads<'a> {
    pub dlogits: &'a mut DeviceBuffer<f32>,
    pub d_embedding_residual: &'a mut DeviceBuffer<f32>,
    pub d_xsa_alphas: &'a mut DeviceBuffer<f32>,
    pub blocks: [BlockBackwardGrads<'a>; GPT2_N_LAYER],
    pub final_norm: LayerNormGrads<'a>,
}
