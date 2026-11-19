use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::attention::{CausalAttentionBackwardTcScratch, CausalAttentionTcScratch};

use super::scratch::AttentionCoreScratch;
use crate::{GPT2_BATCH_SIZE, GPT2_N_EMBD, GPT2_N_HEAD, GPT2_N_LAYER, GPT2_SEQ_LEN};

pub struct AttentionCoreScratchBuffers {
    softmax_d: DeviceBuffer<f32>,
    qk_norm_max: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    backward_q: DeviceBuffer<u16>,
    backward_k: DeviceBuffer<u16>,
    backward_v: DeviceBuffer<u16>,
    backward_d_out: DeviceBuffer<u16>,
    d_out: DeviceBuffer<f32>,
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

impl AttentionCoreScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let compact = GPT2_BATCH_SIZE * GPT2_SEQ_LEN * GPT2_N_EMBD;
        let square = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN * GPT2_SEQ_LEN;
        let batch_head_rows = GPT2_BATCH_SIZE * GPT2_N_HEAD * GPT2_SEQ_LEN;
        Self::allocate(
            stream,
            compact,
            square,
            batch_head_rows,
            2 * GPT2_N_LAYER * GPT2_N_HEAD,
        )
    }

    /// The 64×64 KDA chunk matrices and states are each as large as one
    /// compact head tensor. Forward W and AQK fit together in its F32 arena
    /// as two disjoint half tensors.
    pub fn for_kda(
        stream: &CudaStream,
        batch: u32,
        seq_len: u32,
        heads: u32,
    ) -> Result<Self, DriverError> {
        let head_dim = crate::KDA_CHUNK_SIZE as u32;
        let head_rows = batch
            .checked_mul(seq_len)
            .and_then(|rows| rows.checked_mul(heads))
            .filter(|rows| *rows != 0 && seq_len.is_multiple_of(head_dim))
            .ok_or(DriverError(1))?;
        let compact = head_rows.checked_mul(head_dim).ok_or(DriverError(1))? as usize;
        Self::allocate(
            stream,
            compact,
            compact,
            head_rows as usize,
            2 * heads as usize,
        )
    }

    fn allocate(
        stream: &CudaStream,
        compact: usize,
        matrix_storage: usize,
        batch_head_rows: usize,
        qk_norm_count: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            softmax_d: DeviceBuffer::zeroed(stream, batch_head_rows)?,
            qk_norm_max: DeviceBuffer::zeroed(stream, qk_norm_count)?,
            q: DeviceBuffer::zeroed(stream, compact)?,
            k: DeviceBuffer::zeroed(stream, compact)?,
            v: DeviceBuffer::zeroed(stream, compact)?,
            backward_q: DeviceBuffer::zeroed(stream, compact)?,
            backward_k: DeviceBuffer::zeroed(stream, compact)?,
            backward_v: DeviceBuffer::zeroed(stream, compact)?,
            backward_d_out: DeviceBuffer::zeroed(stream, compact)?,
            d_out: DeviceBuffer::zeroed(stream, compact)?,
            scores: DeviceBuffer::zeroed(stream, matrix_storage)?,
            dot: DeviceBuffer::zeroed(stream, compact)?,
            p: DeviceBuffer::zeroed(stream, compact)?,
            ds: DeviceBuffer::zeroed(stream, compact)?,
            p_half: DeviceBuffer::zeroed(stream, matrix_storage)?,
            ds_half: DeviceBuffer::zeroed(stream, matrix_storage)?,
            d_q: DeviceBuffer::zeroed(stream, compact)?,
            d_k: DeviceBuffer::zeroed(stream, compact)?,
            d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_q: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_k: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_v: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_g: DeviceBuffer::zeroed(stream, compact)?,
            kda_d_beta: DeviceBuffer::zeroed(stream, batch_head_rows)?,
        })
    }

    pub fn args(&mut self) -> AttentionCoreScratch<'_> {
        AttentionCoreScratch {
            softmax_d: &mut self.softmax_d,
            qk_norm_max: &mut self.qk_norm_max,
            tc: CausalAttentionBackwardTcScratch {
                q_f32: &mut self.q,
                k_f32: &mut self.k,
                v_f32: &mut self.v,
                g_f32: &mut self.d_out,
                q: &mut self.backward_q,
                k: &mut self.backward_k,
                v: &mut self.backward_v,
                d_out: &mut self.backward_d_out,
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
            },
        }
    }

    pub fn qk_norm_max(&self) -> &DeviceBuffer<f32> {
        &self.qk_norm_max
    }

    pub fn forward_tc(&mut self) -> CausalAttentionTcScratch<'_> {
        CausalAttentionTcScratch {
            q: &mut self.q,
            k: &mut self.k,
            v: &mut self.v,
            scores: &mut self.scores,
            probs: &mut self.p,
            probs_half: &mut self.p_half,
            compact_out: &mut self.d_out,
            chunk_states: &mut self.backward_q,
        }
    }
}
