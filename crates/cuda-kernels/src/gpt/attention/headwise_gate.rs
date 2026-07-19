use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::AttentionModule;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{abs_f32, exp_f32, max_f32};
use crate::launch::linear_config;
use crate::warp_reduce::warp_sum_f32;

const THREADS_PER_BLOCK: u32 = 256;

pub struct HeadwiseAttentionGateForwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub qkv_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub raw_out_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub gate_offset: u32,
}

pub struct HeadwiseAttentionGateBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv_f16: &'a DeviceBuffer<u16>,
    pub raw_out_f16: &'a DeviceBuffer<u16>,
    pub d_out: &'out mut DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_qkv_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub gate_amax_offset: u32,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub gate_offset: u32,
}

impl AttentionModule {
    pub fn headwise_attention_gate_forward(
        &self,
        args: HeadwiseAttentionGateForwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.embedding_dim, args.head_count * args.head_dim);
        assert!(args.gate_offset + args.head_count <= args.qkv_dim);
        let element_count = args.row_count * args.embedding_dim;
        let config = linear_config(element_count, THREADS_PER_BLOCK);
        match (args.qkv_f16, args.raw_out_f16) {
            (Some(qkv_f16), Some(raw_out_f16)) => {
                assert!(qkv_f16.len() >= (args.row_count * args.qkv_dim) as usize);
                assert!(raw_out_f16.len() >= element_count as usize);
                self.headwise_gate
                    .headwise_attention_gate_forward_save_f16_kernel(
                        args.stream,
                        config,
                        args.qkv,
                        args.out,
                        qkv_f16,
                        raw_out_f16,
                        args.row_count,
                        args.embedding_dim,
                        args.qkv_dim,
                        args.head_count,
                        args.head_dim,
                        args.gate_offset,
                    )
            }
            (None, None) => self.headwise_gate.headwise_attention_gate_forward_kernel(
                args.stream,
                config,
                args.qkv,
                args.out,
                args.row_count,
                args.embedding_dim,
                args.qkv_dim,
                args.head_count,
                args.head_dim,
                args.gate_offset,
            ),
            _ => panic!("headwise gate training tape must save both QKV logits and raw output"),
        }
    }

    pub fn headwise_attention_gate_backward(
        &self,
        args: HeadwiseAttentionGateBackwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        assert_eq!(args.embedding_dim, args.head_count * args.head_dim);
        assert!(args.gate_offset + args.head_count <= args.qkv_dim);
        assert!(args.qkv_f16.len() >= (args.row_count * args.qkv_dim) as usize);
        assert!(args.raw_out_f16.len() >= (args.row_count * args.embedding_dim) as usize);
        assert!(args.d_qkv_chunk_amax.len() >= (args.gate_amax_offset + args.row_count) as usize);
        self.headwise_gate.headwise_attention_gate_backward_kernel(
            args.stream,
            linear_config(args.row_count * THREADS_PER_BLOCK, THREADS_PER_BLOCK),
            args.qkv_f16,
            args.raw_out_f16,
            args.d_out,
            args.d_qkv,
            args.d_qkv_chunk_amax,
            args.gate_amax_offset,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
            args.head_count,
            args.head_dim,
            args.gate_offset,
        )
    }
}

#[inline(always)]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-value))
}

#[inline(always)]
fn gate_position(
    element_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_dim: u32,
    gate_offset: u32,
) -> Option<(usize, usize, u32)> {
    let index = thread::blockIdx_x() * THREADS_PER_BLOCK + thread::threadIdx_x();
    if index >= element_count {
        return None;
    }
    let row = index / embedding_dim;
    let col = index - row * embedding_dim;
    let head = col / head_dim;
    let gate_index = row * qkv_dim + gate_offset + head;
    Some((index as usize, gate_index as usize, col % head_dim))
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit shape fields"
)]
fn forward_body(
    qkv: &[f32],
    mut out: DisjointSlice<f32>,
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    _head_count: u32,
    head_dim: u32,
    gate_offset: u32,
) {
    let Some((index, gate_index, _)) = gate_position(
        row_count * embedding_dim,
        embedding_dim,
        qkv_dim,
        head_dim,
        gate_offset,
    ) else {
        return;
    };
    unsafe {
        *out.get_unchecked_mut(index) *= sigmoid(qkv[gate_index]);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit shape fields"
)]
fn forward_save_body(
    qkv: &[f32],
    mut out: DisjointSlice<f32>,
    mut qkv_f16: DisjointSlice<u16>,
    mut raw_out_f16: DisjointSlice<u16>,
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    _head_count: u32,
    head_dim: u32,
    gate_offset: u32,
) {
    let Some((index, gate_index, head_dim_index)) = gate_position(
        row_count * embedding_dim,
        embedding_dim,
        qkv_dim,
        head_dim,
        gate_offset,
    ) else {
        return;
    };
    let raw = unsafe { *out.as_mut_ptr().add(index) };
    let gate_logit = qkv[gate_index];
    unsafe {
        *out.get_unchecked_mut(index) = raw * sigmoid(gate_logit);
        *raw_out_f16.get_unchecked_mut(index) = cvt_rn_f16_f32(raw);
        if head_dim_index == 0 {
            *qkv_f16.get_unchecked_mut(gate_index) = cvt_rn_f16_f32(gate_logit);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit shape fields"
)]
fn backward_body(
    qkv_f16: &[u16],
    raw_out_f16: &[u16],
    mut d_out: DisjointSlice<f32>,
    mut d_qkv: DisjointSlice<f32>,
    mut d_qkv_chunk_amax: DisjointSlice<f32>,
    gate_amax_offset: u32,
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
    gate_offset: u32,
    gate_warp_amax: &mut SharedArray<f32, 8>,
) {
    let row = thread::blockIdx_x();
    if row >= row_count {
        return;
    }
    let thread_id = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_in_block = thread_id / 32;
    let mut warp_amax = 0.0;
    let mut head = warp_in_block;
    while head < head_count {
        let gate_index = (row * qkv_dim + gate_offset + head) as usize;
        let gate = warp::shuffle_f32(
            if lane == 0 {
                sigmoid(cvt_f32_f16(qkv_f16[gate_index]))
            } else {
                0.0
            },
            0,
        );
        let mut dot = 0.0;
        let mut dim = lane;
        while dim < head_dim {
            let hidden_index = (row * embedding_dim + head * head_dim + dim) as usize;
            let grad = unsafe { *d_out.as_mut_ptr().add(hidden_index) };
            dot += grad * cvt_f32_f16(raw_out_f16[hidden_index]);
            unsafe {
                *d_out.get_unchecked_mut(hidden_index) = grad * gate;
            }
            dim += 32;
        }
        let gate_grad = warp_sum_f32(dot) * gate * (1.0 - gate);
        if lane == 0 {
            unsafe {
                *d_qkv.get_unchecked_mut(gate_index) = gate_grad;
            }
            warp_amax = max_f32(warp_amax, abs_f32(gate_grad));
        }
        head += THREADS_PER_BLOCK / 32;
    }
    if lane == 0 {
        gate_warp_amax[warp_in_block as usize] = warp_amax;
    }
    thread::sync_threads();
    if thread_id == 0 {
        let mut row_amax = 0.0;
        let mut warp_index = 0;
        while warp_index < 8 {
            row_amax = max_f32(row_amax, gate_warp_amax[warp_index]);
            warp_index += 1;
        }
        unsafe {
            *d_qkv_chunk_amax.get_unchecked_mut((gate_amax_offset + row) as usize) = row_amax;
        }
    }
    let mut padding_col = gate_offset + head_count + thread_id;
    while padding_col < qkv_dim {
        unsafe {
            *d_qkv.get_unchecked_mut((row * qkv_dim + padding_col) as usize) = 0.0;
        }
        padding_col += THREADS_PER_BLOCK;
    }
}

#[cuda_module]
pub(super) mod kernels {
    use super::*;

    #[kernel]
    pub fn headwise_attention_gate_forward_kernel(
        qkv: &[f32],
        out: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        gate_offset: u32,
    ) {
        forward_body(
            qkv,
            out,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            gate_offset,
        );
    }

    #[kernel]
    pub fn headwise_attention_gate_forward_save_f16_kernel(
        qkv: &[f32],
        out: DisjointSlice<f32>,
        qkv_f16: DisjointSlice<u16>,
        raw_out_f16: DisjointSlice<u16>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        gate_offset: u32,
    ) {
        forward_save_body(
            qkv,
            out,
            qkv_f16,
            raw_out_f16,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            gate_offset,
        );
    }

    #[kernel]
    pub fn headwise_attention_gate_backward_kernel(
        qkv_f16: &[u16],
        raw_out_f16: &[u16],
        d_out: DisjointSlice<f32>,
        d_qkv: DisjointSlice<f32>,
        d_qkv_chunk_amax: DisjointSlice<f32>,
        gate_amax_offset: u32,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        gate_offset: u32,
    ) {
        static mut GATE_WARP_AMAX: SharedArray<f32, 8> = SharedArray::UNINIT;
        backward_body(
            qkv_f16,
            raw_out_f16,
            d_out,
            d_qkv,
            d_qkv_chunk_amax,
            gate_amax_offset,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            gate_offset,
            unsafe { &mut GATE_WARP_AMAX },
        );
    }
}
