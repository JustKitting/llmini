use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{CanonTensors, CanonWeights};

use crate::AppResult;

pub struct UploadedCanon {
    pub(crate) weight: DeviceBuffer<f32>,
}

impl UploadedCanon {
    pub(in crate::upload) fn new(stream: &CudaStream, weights: &CanonWeights) -> AppResult<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(stream, &weights.values)?,
        })
    }

    pub fn tensors(&self) -> CanonTensors<'_> {
        CanonTensors {
            weight: &self.weight,
        }
    }

    pub(crate) fn zero(stream: &CudaStream) -> AppResult<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(
                stream,
                &vec![0.0; gpt2_nvfp4::GPT2_CANON_WEIGHT_COUNT],
            )?,
        })
    }

    pub(crate) fn from_host(stream: &CudaStream, weight: &[f32]) -> AppResult<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(stream, weight)?,
        })
    }
}
