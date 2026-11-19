use cuda_core::CudaStream;
use gpt2_nvfp4::{LayerNormTensors, LayerNormWeights, LinearWeights, Nvfp4Shape};

use super::{UploadedNvfp4, upload_nvfp4};
use crate::AppResult;

pub struct UploadedPair {
    pub(crate) weight: UploadedNvfp4,
    pub(crate) bias: UploadedNvfp4,
}

pub type UploadedLayerNorm = UploadedPair;
pub type UploadedLinear = UploadedPair;

impl UploadedPair {
    pub(crate) fn from_parts(weight: UploadedNvfp4, bias: UploadedNvfp4) -> Self {
        Self { weight, bias }
    }

    pub(in crate::upload) fn from_layer_norm(
        stream: &CudaStream,
        layer_norm: &LayerNormWeights,
    ) -> AppResult<Self> {
        Ok(Self {
            weight: upload_nvfp4(stream, &layer_norm.weight)?,
            bias: upload_nvfp4(stream, &layer_norm.bias)?,
        })
    }

    pub(in crate::upload) fn from_linear<W: Nvfp4Shape, B: Nvfp4Shape>(
        stream: &CudaStream,
        linear: &LinearWeights<W, B>,
    ) -> AppResult<Self> {
        Ok(Self {
            weight: super::nvfp4::upload_nvfp4_k(stream, &linear.weight, W::ROWS)?,
            bias: upload_nvfp4(stream, &linear.bias)?,
        })
    }

    pub fn tensors(&self) -> LayerNormTensors<'_> {
        LayerNormTensors {
            weight: self.weight.device(),
            bias: self.bias.device(),
        }
    }
}
