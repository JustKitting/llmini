use crate::random::InitRng;
use crate::types::{AttentionWeights, CanonWeights, LayerNormWeights, MlpWeights};

#[derive(Clone, Debug)]
pub struct Gpt2BlockWeights {
    pub ln_1: LayerNormWeights,
    pub canon_a: CanonWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub canon_c: CanonWeights,
    pub mlp: MlpWeights,
}

impl Gpt2BlockWeights {
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        Self {
            ln_1: LayerNormWeights::init(),
            canon_a: CanonWeights::init(rng),
            attn: AttentionWeights::init(rng, residual_projection_scale),
            ln_2: LayerNormWeights::init(),
            canon_c: CanonWeights::init(rng),
            mlp: MlpWeights::init(rng, residual_projection_scale),
        }
    }
}
