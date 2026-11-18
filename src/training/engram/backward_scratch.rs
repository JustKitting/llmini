use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_TOKEN_ROWS, LinearScratch};

// Engram insertions execute serially on Runtime::stream. Their forward tapes
// and parameter gradients remain separate; these temporaries die at the end of
// each insertion's backward call and can therefore share one allocation set.
pub(super) struct BackwardScratch {
    pub linear: LinearScratch,
    pub d_un: DeviceBuffer<f32>,
    pub d_u: DeviceBuffer<f32>,
    pub d_key: DeviceBuffer<f32>,
    pub d_value: DeviceBuffer<f32>,
    pub d_dot: DeviceBuffer<f32>,
    pub d_mem_key: DeviceBuffer<f32>,
    pub d_mem_value: DeviceBuffer<f32>,
}

impl BackwardScratch {
    pub fn new(s: &CudaStream, memory: usize, streams: usize) -> Result<Self, DriverError> {
        let single = GPT2_TOKEN_ROWS * GPT2_N_EMBD;
        let total = single * streams;
        let mem = GPT2_TOKEN_ROWS * memory;
        Ok(Self {
            linear: LinearScratch::new(s, memory, GPT2_N_EMBD)?,
            d_un: DeviceBuffer::zeroed(s, total)?,
            d_u: DeviceBuffer::zeroed(s, total)?,
            d_key: DeviceBuffer::zeroed(s, total)?,
            d_value: DeviceBuffer::zeroed(s, single)?,
            d_dot: DeviceBuffer::zeroed(s, GPT2_TOKEN_ROWS * streams)?,
            d_mem_key: DeviceBuffer::zeroed(s, mem)?,
            d_mem_value: DeviceBuffer::zeroed(s, mem)?,
        })
    }
}
