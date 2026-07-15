use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::CausalAttentionBackwardTcScratch;

use super::shape::{HEAD_DIM, HEADS, TOKEN_COUNT};

pub struct TcScratchBuffers {
    q_f32: DeviceBuffer<f32>,
    k_f32: DeviceBuffer<f32>,
    v_f32: DeviceBuffer<f32>,
    g_f32: DeviceBuffer<f32>,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    d_out: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    dot: DeviceBuffer<f32>,
    p: DeviceBuffer<f32>,
    ds: DeviceBuffer<f32>,
    p_half: DeviceBuffer<u16>,
    ds_half: DeviceBuffer<u16>,
    d_q: DeviceBuffer<f32>,
    d_k: DeviceBuffer<f32>,
    d_v: DeviceBuffer<f32>,
    kda_d_q: DeviceBuffer<f32>,
    kda_d_k: DeviceBuffer<f32>,
    kda_d_v: DeviceBuffer<f32>,
    kda_d_g: DeviceBuffer<f32>,
    kda_d_beta: DeviceBuffer<f32>,
}

impl TcScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = HEADS * TOKEN_COUNT * HEAD_DIM;
        let square = HEADS * TOKEN_COUNT * TOKEN_COUNT;
        Ok(Self {
            q_f32: DeviceBuffer::zeroed(stream, compact)?,
            k_f32: DeviceBuffer::zeroed(stream, compact)?,
            v_f32: DeviceBuffer::zeroed(stream, compact)?,
            g_f32: DeviceBuffer::zeroed(stream, compact)?,
            q: DeviceBuffer::zeroed(stream, compact)?,
            k: DeviceBuffer::zeroed(stream, compact)?,
            v: DeviceBuffer::zeroed(stream, compact)?,
            d_out: DeviceBuffer::zeroed(stream, compact)?,
            scores: DeviceBuffer::zeroed(stream, square)?,
            dot: DeviceBuffer::zeroed(stream, square)?,
            p: DeviceBuffer::zeroed(stream, compact)?,
            ds: DeviceBuffer::zeroed(stream, compact)?,
            p_half: DeviceBuffer::zeroed(stream, square)?,
            ds_half: DeviceBuffer::zeroed(stream, square)?,
            d_q: DeviceBuffer::zeroed(stream, compact)?,
            d_k: DeviceBuffer::zeroed(stream, compact)?,
            d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_q: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_k: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_g: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_beta: DeviceBuffer::zeroed(stream, TOKEN_COUNT * HEADS)?,
        })
    }

    pub fn args(&mut self) -> CausalAttentionBackwardTcScratch<'_> {
        CausalAttentionBackwardTcScratch {
            q_f32: &mut self.q_f32,
            k_f32: &mut self.k_f32,
            v_f32: &mut self.v_f32,
            g_f32: &mut self.g_f32,
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            d_out: &mut self.d_out,
            scores: &mut self.scores,
            dot: &mut self.dot,
            p: &mut self.p,
            ds: &mut self.ds,
            p_half: &mut self.p_half,
            ds_half: &mut self.ds_half,
            d_q: &mut self.d_q,
            d_k: &mut self.d_k,
            d_v: &mut self.d_v,
            kda_d_q: &mut self.kda_d_q,
            kda_d_k: &mut self.kda_d_k,
            kda_d_v: &mut self.kda_d_v,
            kda_d_g: &mut self.kda_d_g,
            kda_d_beta: &mut self.kda_d_beta,
        }
    }

    pub fn set_forward_probs(&mut self, probs: DeviceBuffer<u16>) {
        self.p_half = probs;
    }
}
