use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_VOCAB_SIZE};
use rust_kernels_cuda::nvfp4_quant::nvfp4_tensor_amax_chunks;

pub struct OptimizerScratch {
    pub(super) amax: DeviceBuffer<f32>,
    pub(super) chunk_amax: DeviceBuffer<f32>,
    pub(super) kda_clip_scores: DeviceBuffer<f32>,
    pub(super) kda_clip_factors: DeviceBuffer<f32>,
}

impl OptimizerScratch {
    pub(in crate::training) fn ensure_nvfp4_capacity(
        &mut self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<(), DriverError> {
        let count = nvfp4_tensor_amax_chunks(len);
        if self.chunk_amax.len() < count {
            self.chunk_amax = DeviceBuffer::zeroed(stream, count)?;
        }
        Ok(())
    }
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            amax: DeviceBuffer::zeroed(stream, 1)?,
            chunk_amax: DeviceBuffer::zeroed(
                stream,
                nvfp4_tensor_amax_chunks(GPT2_VOCAB_SIZE * GPT2_N_EMBD),
            )?,
            kda_clip_scores: DeviceBuffer::zeroed(stream, GPT2_N_HEAD)?,
            kda_clip_factors: DeviceBuffer::zeroed(stream, GPT2_N_LAYER * GPT2_N_HEAD)?,
        })
    }
}
