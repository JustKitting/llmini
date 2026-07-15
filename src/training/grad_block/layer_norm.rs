use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, LayerNormGrads};

use crate::training::device_buffer::zero;

pub struct LayerNormGradBuffers {
    pub(in crate::training) d_weight: DeviceBuffer<f32>,
    pub(in crate::training) d_bias: DeviceBuffer<f32>,
}

impl LayerNormGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_weight: zero(stream, GPT2_N_EMBD)?,
            d_bias: zero(stream, GPT2_N_EMBD)?,
        })
    }

    pub fn grads(&mut self) -> LayerNormGrads<'_> {
        LayerNormGrads {
            d_weight: &mut self.d_weight,
            d_bias: &mut self.d_bias,
        }
    }
}
