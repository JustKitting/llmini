#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::CausalAttentionTcScratch;

pub struct CausalAttentionTcScratchBuffers {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    probs_half: DeviceBuffer<u16>,
    compact_out: DeviceBuffer<f32>,
    chunk_states: DeviceBuffer<u16>,
}

impl CausalAttentionTcScratchBuffers {
    pub fn new(
        stream: &CudaStream,
        compact_len: usize,
        batch_size: usize,
        head_count: usize,
        seq_len: usize,
    ) -> Result<Self, DriverError> {
        let square = batch_size * head_count * seq_len * seq_len;
        Ok(Self {
            q: DeviceBuffer::zeroed(stream, compact_len)?,
            k: DeviceBuffer::zeroed(stream, compact_len)?,
            v: DeviceBuffer::zeroed(stream, compact_len)?,
            scores: DeviceBuffer::zeroed(stream, square)?,
            probs: DeviceBuffer::zeroed(stream, compact_len)?,
            probs_half: DeviceBuffer::zeroed(stream, square)?,
            compact_out: DeviceBuffer::zeroed(stream, compact_len)?,
            chunk_states: DeviceBuffer::zeroed(stream, compact_len)?,
        })
    }

    pub fn args(&mut self) -> CausalAttentionTcScratch<'_> {
        CausalAttentionTcScratch {
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            scores: &mut self.scores,
            probs: &mut self.probs,
            probs_half: &mut self.probs_half,
            compact_out: &mut self.compact_out,
            chunk_states: &mut self.chunk_states,
        }
    }
}
