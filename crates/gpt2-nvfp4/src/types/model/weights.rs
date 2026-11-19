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
    pub(crate) fn init(rng: &mut InitRng, depth_growth: bool) -> Self {
        let mut residual_projection_scale = 0.02;
        if depth_growth {
            // LIDAS initializes residual projections against the starting depth.
            residual_projection_scale /= (2.0 * crate::KIMI_FULL_ATTENTION_PERIOD as f32).sqrt();
        }
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

    pub fn forward_with_hook<'a>(
        &self,
        args: Gpt2ForwardArgs<'a>,
        hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward_with_hook(self, args, hook)
    }
}

#[cfg(test)]
mod tests {
    use crate::Gpt2;

    #[test]
    fn depth_growth_initializes_residual_projections_at_the_first_stage_depth() {
        let mut fixed = Gpt2::new();
        let mut grown = Gpt2::new();
        fixed.init(0x47505432);
        grown.init_for_depth_growth(0x47505432);
        let fixed = &fixed.weights().unwrap().h;
        let grown = &grown.weights().unwrap().h;
        let expected = 0.02 / 8.0_f32.sqrt();
        for (fixed, grown) in fixed.iter().zip(grown) {
            assert_eq!(grown.attn.c_proj.weight.global_scale, expected);
            assert_eq!(grown.mlp.c_proj.weight.global_scale, expected);
            assert_eq!(fixed.attn.c_proj.weight.global_scale, 0.02);
            assert_eq!(fixed.mlp.c_proj.weight.global_scale, 0.02);
            assert_eq!(
                grown.attn.c_attn.weight.global_scale,
                fixed.attn.c_attn.weight.global_scale
            );
            assert_eq!(
                grown.mlp.c_fc.weight.global_scale,
                fixed.mlp.c_fc.weight.global_scale
            );
            assert!(
                grown.attn.c_proj.weight.bytes.as_ref() == fixed.attn.c_proj.weight.bytes.as_ref()
            );
            assert!(
                grown.mlp.c_proj.weight.bytes.as_ref() == fixed.mlp.c_proj.weight.bytes.as_ref()
            );
            assert!(
                grown.attn.c_proj.weight.scales.as_ref()
                    == fixed.attn.c_proj.weight.scales.as_ref()
            );
            assert!(
                grown.mlp.c_proj.weight.scales.as_ref() == fixed.mlp.c_proj.weight.scales.as_ref()
            );
        }
    }
}
