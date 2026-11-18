use crate::random::InitRng;
use crate::{Nvfp4Shape, Nvfp4Tensor};

use super::shapes::Nvfp4ShapeInit;

#[derive(Clone, Debug)]
pub struct LinearWeights<W: Nvfp4Shape, B: Nvfp4Shape> {
    pub weight: Nvfp4Tensor<W>,
    pub bias: Nvfp4Tensor<B>,
}

impl<W: Nvfp4Shape, B: Nvfp4Shape> LinearWeights<W, B> {
    pub(crate) fn init(rng: &mut InitRng) -> Self
    where
        W: Nvfp4ShapeInit,
        B: Nvfp4ShapeInit,
    {
        Self {
            weight: W::smooth_tensor(rng),
            bias: B::zero_tensor(),
        }
    }

    pub(crate) fn init_with_weight_scale(rng: &mut InitRng, weight_scale: f32) -> Self
    where
        W: Nvfp4ShapeInit,
        B: Nvfp4ShapeInit,
    {
        Self {
            weight: W::smooth_tensor_with_global_scale(rng, weight_scale),
            bias: B::zero_tensor(),
        }
    }

    pub(crate) fn init_reference_normal(rng: &mut InitRng, weight_scale: f32) -> Self
    where
        W: Nvfp4ShapeInit,
        B: Nvfp4ShapeInit,
    {
        Self {
            weight: W::reference_normal_tensor(rng, weight_scale),
            bias: B::zero_tensor(),
        }
    }
}
