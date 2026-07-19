use cuda_core::DriverError;

use super::args::Gpt2ForwardArgs;
use super::forward;
use crate::random::InitRng;
use crate::types::{
    EmbeddingWeights, Gpt2BlockWeights, HiddenStateDevice, LayerNormWeights, NextLatWeights,
    Nvfp4ShapeInit, XsaAlphaShape, XsaAlphaTensor,
};
use crate::{GPT2_N_LAYER, Gpt2Config};

#[derive(Clone, Debug)]
pub struct Gpt2Weights {
    pub config: Gpt2Config,
    pub embeddings: EmbeddingWeights,
    pub h: [Gpt2BlockWeights; GPT2_N_LAYER],
    pub ln_f: LayerNormWeights,
    pub next_latent: NextLatWeights,
    pub xsa_alphas: XsaAlphaTensor,
}

impl Gpt2Weights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        let residual_projection_scale = 0.02;
        Self {
            config: Gpt2Config::gpt2_124m(),
            embeddings: EmbeddingWeights::init(rng),
            h: std::array::from_fn(|_| Gpt2BlockWeights::init(rng, residual_projection_scale)),
            ln_f: LayerNormWeights::init(),
            next_latent: NextLatWeights::init(rng),
            xsa_alphas: XsaAlphaShape::zero_tensor(),
        }
    }

    pub fn forward_embeddings<'a>(
        &self,
        args: crate::types::TokenEmbeddingArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        self.embeddings.forward(args)
    }

    pub fn forward<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(self, args)
    }
}
