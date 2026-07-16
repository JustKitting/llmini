use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, HiddenState, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

use super::super::device_buffer::zero;

pub struct NextLatGradBuffers {
    // Ping buffer: d_act2 -> d_act1 -> d_normalized.
    pub d_act2: DeviceBuffer<f32>,
    // Pong buffer: d_pre2 -> d_pre1 -> d_concat.
    pub d_pre2: DeviceBuffer<f32>,
    pub d_next_token_embeddings: DeviceBuffer<f32>,
    pub d_current_states: DeviceBuffer<f32>,
    pub d_norm_weight: DeviceBuffer<f32>,
    pub d_norm_bias: DeviceBuffer<f32>,
    pub d_input_projection_weight: DeviceBuffer<f32>,
    pub d_input_projection_bias: DeviceBuffer<f32>,
    pub d_transition_weight: DeviceBuffer<f32>,
    pub d_transition_bias: DeviceBuffer<f32>,
    pub d_output_projection_weight: DeviceBuffer<f32>,
    pub d_output_projection_bias: DeviceBuffer<f32>,
}

impl NextLatGradBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            d_act2: zero(stream, gpt2_nvfp4::NextLatHiddenActivation::LEN)?,
            d_pre2: zero(stream, gpt2_nvfp4::NextLatHiddenActivation::LEN)?,
            d_next_token_embeddings: zero(stream, HiddenState::LEN)?,
            d_current_states: zero(stream, HiddenState::LEN)?,
            d_norm_weight: zero(stream, NEXTLAT_INPUT)?,
            d_norm_bias: zero(stream, NEXTLAT_INPUT)?,
            d_input_projection_weight: zero(stream, NEXTLAT_INPUT * NEXTLAT_HIDDEN)?,
            d_input_projection_bias: zero(stream, NEXTLAT_HIDDEN)?,
            d_transition_weight: zero(stream, NEXTLAT_HIDDEN * NEXTLAT_HIDDEN)?,
            d_transition_bias: zero(stream, NEXTLAT_HIDDEN)?,
            d_output_projection_weight: zero(stream, NEXTLAT_HIDDEN * GPT2_N_EMBD)?,
            d_output_projection_bias: zero(stream, GPT2_N_EMBD)?,
        })
    }
}
