use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::AttentionModule;
use crate::atomic::atomic_add_f32;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::{abs_f32, exp_f32, max_f32, sqrt_f32};
use crate::kda_common::{silu, silu_grad};
use crate::launch::{launch_config, linear_config};
use crate::nvfp4::{Nvfp4DeviceTensor, nvfp4_value};
use crate::warp_reduce::warp_sum_f32;

const THREADS_PER_BLOCK: u32 = 256;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / 32;
const XSA_MAX_ROW_TILES: u32 = 128;
const XSA_NORM_EPS: f32 = 1.0e-4;

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

pub struct ExclusiveSelfAttentionForwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub xsa_alphas: Nvfp4DeviceTensor<'a>,
    pub qkv_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub raw_out_f16: Option<&'out mut DeviceBuffer<u16>>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub value_offset: u32,
    pub gate_offset: u32,
    pub alpha_offset: u32,
    pub apply_headwise_gate: bool,
    pub kda_value_activation: bool,
}

pub struct ExclusiveSelfAttentionBackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub qkv_f16: &'a DeviceBuffer<u16>,
    pub raw_out_f16: &'a DeviceBuffer<u16>,
    pub d_out: &'out mut DeviceBuffer<f32>,
    pub d_qkv: &'out mut DeviceBuffer<f32>,
    pub d_xsa_alphas: &'out mut DeviceBuffer<f32>,
    pub xsa_alphas: Nvfp4DeviceTensor<'a>,
    pub row_count: u32,
    pub embedding_dim: u32,
    pub qkv_dim: u32,
    pub head_count: u32,
    pub head_dim: u32,
    pub value_offset: u32,
    pub gate_offset: u32,
    pub alpha_offset: u32,
    pub apply_headwise_gate: bool,
    pub kda_value_activation: bool,
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

    pub fn exclusive_self_attention_forward(
        &self,
        args: ExclusiveSelfAttentionForwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        validate_xsa_shape(
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
            args.head_count,
            args.head_dim,
            args.value_offset,
            args.gate_offset,
            args.alpha_offset,
            args.apply_headwise_gate,
            args.qkv.len(),
            args.out.len(),
            args.xsa_alphas,
        );
        let config = xsa_config(args.row_count, args.head_count);
        let apply_headwise_gate = u32::from(args.apply_headwise_gate);
        let kda_value_activation = u32::from(args.kda_value_activation);
        match (args.qkv_f16, args.raw_out_f16) {
            (Some(qkv_f16), Some(raw_out_f16)) => {
                assert!(qkv_f16.len() >= (args.row_count * args.qkv_dim) as usize);
                assert!(raw_out_f16.len() >= (args.row_count * args.embedding_dim) as usize);
                self.headwise_gate
                    .exclusive_self_attention_forward_save_f16_kernel(
                        args.stream,
                        config,
                        args.qkv,
                        args.out,
                        args.xsa_alphas.bytes,
                        args.xsa_alphas.scales,
                        args.xsa_alphas.global_scale,
                        qkv_f16,
                        raw_out_f16,
                        args.row_count,
                        args.embedding_dim,
                        args.qkv_dim,
                        args.head_count,
                        args.head_dim,
                        args.value_offset,
                        args.gate_offset,
                        args.alpha_offset,
                        apply_headwise_gate,
                        kda_value_activation,
                    )
            }
            (None, None) => self.headwise_gate.exclusive_self_attention_forward_kernel(
                args.stream,
                config,
                args.qkv,
                args.out,
                args.xsa_alphas.bytes,
                args.xsa_alphas.scales,
                args.xsa_alphas.global_scale,
                args.row_count,
                args.embedding_dim,
                args.qkv_dim,
                args.head_count,
                args.head_dim,
                args.value_offset,
                args.gate_offset,
                args.alpha_offset,
                apply_headwise_gate,
                kda_value_activation,
            ),
            _ => panic!("XSA training tape must save both QKV logits and raw output"),
        }
    }

    pub fn exclusive_self_attention_backward(
        &self,
        args: ExclusiveSelfAttentionBackwardArgs<'_, '_>,
    ) -> Result<(), DriverError> {
        validate_xsa_shape(
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
            args.head_count,
            args.head_dim,
            args.value_offset,
            args.gate_offset,
            args.alpha_offset,
            args.apply_headwise_gate,
            args.qkv_f16.len(),
            args.d_out.len(),
            args.xsa_alphas,
        );
        assert!(args.raw_out_f16.len() >= (args.row_count * args.embedding_dim) as usize);
        assert!(args.d_qkv.len() >= (args.row_count * args.qkv_dim) as usize);
        assert!(args.d_xsa_alphas.len() >= (args.alpha_offset + args.head_count) as usize);
        self.headwise_gate.exclusive_self_attention_backward_kernel(
            args.stream,
            xsa_config(args.row_count, args.head_count),
            args.qkv_f16,
            args.raw_out_f16,
            args.d_out,
            args.d_qkv,
            args.d_xsa_alphas,
            args.xsa_alphas.bytes,
            args.xsa_alphas.scales,
            args.xsa_alphas.global_scale,
            args.row_count,
            args.embedding_dim,
            args.qkv_dim,
            args.head_count,
            args.head_dim,
            args.value_offset,
            args.gate_offset,
            args.alpha_offset,
            u32::from(args.apply_headwise_gate),
            u32::from(args.kda_value_activation),
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "shape validation mirrors the explicit CUDA ABI"
)]
fn validate_xsa_shape(
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
    value_offset: u32,
    gate_offset: u32,
    alpha_offset: u32,
    apply_headwise_gate: bool,
    qkv_len: usize,
    out_len: usize,
    xsa_alphas: Nvfp4DeviceTensor<'_>,
) {
    assert!(row_count > 0);
    assert_eq!(embedding_dim, head_count * head_dim);
    assert!(value_offset + embedding_dim <= qkv_dim);
    if apply_headwise_gate {
        assert!(gate_offset + head_count <= qkv_dim);
    }
    assert!(qkv_len >= (row_count * qkv_dim) as usize);
    assert!(out_len >= (row_count * embedding_dim) as usize);
    let alpha_end = (alpha_offset + head_count) as usize;
    assert!(xsa_alphas.bytes.len() * 2 >= alpha_end);
    assert!(xsa_alphas.scales.len() * 16 >= alpha_end);
    assert!(!xsa_alphas.global_scale.is_empty());
}

fn xsa_config(row_count: u32, head_count: u32) -> LaunchConfig {
    let row_tiles = row_count.div_ceil(WARPS_PER_BLOCK).min(XSA_MAX_ROW_TILES);
    launch_config((row_tiles, head_count, 1), THREADS_PER_BLOCK)
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

#[inline(always)]
fn tanh_gate(value: f32) -> f32 {
    2.0 * sigmoid(2.0 * value) - 1.0
}

#[inline(always)]
fn xsa_alpha(
    alpha_bytes: &[u8],
    alpha_scales: &[u8],
    alpha_global_scale: &[f32],
    alpha_offset: u32,
    head: u32,
) -> f32 {
    tanh_gate(nvfp4_value(
        alpha_bytes,
        alpha_scales,
        alpha_global_scale[0],
        (alpha_offset + head) as usize,
    ))
}

#[inline(always)]
fn xsa_value(raw_value: f32, kda_value_activation: u32) -> f32 {
    if kda_value_activation != 0 {
        silu(raw_value)
    } else {
        raw_value
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit tensor and shape fields"
)]
fn xsa_forward_body(
    qkv: &[f32],
    mut out: DisjointSlice<f32>,
    alpha_bytes: &[u8],
    alpha_scales: &[u8],
    alpha_global_scale: &[f32],
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
    value_offset: u32,
    gate_offset: u32,
    alpha_offset: u32,
    apply_headwise_gate: u32,
    kda_value_activation: u32,
) {
    let head = thread::blockIdx_y();
    if head >= head_count {
        return;
    }
    let lane = warp::lane_id();
    let warp_in_block = thread::threadIdx_x() / 32;
    let row_stride = thread::gridDim_x() * WARPS_PER_BLOCK;
    let mut row = thread::blockIdx_x() * WARPS_PER_BLOCK + warp_in_block;
    let alpha = xsa_alpha(
        alpha_bytes,
        alpha_scales,
        alpha_global_scale,
        alpha_offset,
        head,
    );
    while row < row_count {
        let hidden_base = row * embedding_dim + head * head_dim;
        let value_base = row * qkv_dim + value_offset + head * head_dim;
        let mut value_sumsq = 0.0;
        let mut dim = lane;
        while dim < head_dim {
            let value = xsa_value(qkv[(value_base + dim) as usize], kda_value_activation);
            value_sumsq += value * value;
            dim += 32;
        }
        let value_norm = max_f32(sqrt_f32(warp_sum_f32(value_sumsq)), XSA_NORM_EPS);
        let value_norm_inv = 1.0 / value_norm;

        let mut projection = 0.0;
        dim = lane;
        while dim < head_dim {
            let value =
                xsa_value(qkv[(value_base + dim) as usize], kda_value_activation) * value_norm_inv;
            projection += unsafe { *out.as_mut_ptr().add((hidden_base + dim) as usize) } * value;
            dim += 32;
        }
        let projection = warp_sum_f32(projection);
        let gate = if apply_headwise_gate != 0 {
            sigmoid(qkv[(row * qkv_dim + gate_offset + head) as usize])
        } else {
            1.0
        };

        dim = lane;
        while dim < head_dim {
            let hidden_index = (hidden_base + dim) as usize;
            let value =
                xsa_value(qkv[(value_base + dim) as usize], kda_value_activation) * value_norm_inv;
            let raw = unsafe { *out.as_mut_ptr().add(hidden_index) };
            unsafe {
                *out.get_unchecked_mut(hidden_index) = (raw - alpha * projection * value) * gate;
            }
            dim += 32;
        }
        row += row_stride;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit tensor and shape fields"
)]
fn xsa_forward_save_body(
    qkv: &[f32],
    mut out: DisjointSlice<f32>,
    alpha_bytes: &[u8],
    alpha_scales: &[u8],
    alpha_global_scale: &[f32],
    mut qkv_f16: DisjointSlice<u16>,
    mut raw_out_f16: DisjointSlice<u16>,
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
    value_offset: u32,
    gate_offset: u32,
    alpha_offset: u32,
    apply_headwise_gate: u32,
    kda_value_activation: u32,
) {
    let head = thread::blockIdx_y();
    if head >= head_count {
        return;
    }
    let lane = warp::lane_id();
    let warp_in_block = thread::threadIdx_x() / 32;
    let row_stride = thread::gridDim_x() * WARPS_PER_BLOCK;
    let mut row = thread::blockIdx_x() * WARPS_PER_BLOCK + warp_in_block;
    let alpha = xsa_alpha(
        alpha_bytes,
        alpha_scales,
        alpha_global_scale,
        alpha_offset,
        head,
    );
    while row < row_count {
        let hidden_base = row * embedding_dim + head * head_dim;
        let value_base = row * qkv_dim + value_offset + head * head_dim;
        let mut value_sumsq = 0.0;
        let mut dim = lane;
        while dim < head_dim {
            let value = xsa_value(qkv[(value_base + dim) as usize], kda_value_activation);
            value_sumsq += value * value;
            dim += 32;
        }
        let value_norm = max_f32(sqrt_f32(warp_sum_f32(value_sumsq)), XSA_NORM_EPS);
        let value_norm_inv = 1.0 / value_norm;

        let mut projection = 0.0;
        dim = lane;
        while dim < head_dim {
            let value =
                xsa_value(qkv[(value_base + dim) as usize], kda_value_activation) * value_norm_inv;
            projection += unsafe { *out.as_mut_ptr().add((hidden_base + dim) as usize) } * value;
            dim += 32;
        }
        let projection = warp_sum_f32(projection);
        let gate_index = (row * qkv_dim + gate_offset + head) as usize;
        let gate = if apply_headwise_gate != 0 {
            sigmoid(qkv[gate_index])
        } else {
            1.0
        };

        dim = lane;
        while dim < head_dim {
            let hidden_index = (hidden_base + dim) as usize;
            let value =
                xsa_value(qkv[(value_base + dim) as usize], kda_value_activation) * value_norm_inv;
            let raw = unsafe { *out.as_mut_ptr().add(hidden_index) };
            unsafe {
                *raw_out_f16.get_unchecked_mut(hidden_index) = cvt_rn_f16_f32(raw);
                *out.get_unchecked_mut(hidden_index) = (raw - alpha * projection * value) * gate;
            }
            dim += 32;
        }
        if lane == 0 && apply_headwise_gate != 0 {
            unsafe {
                *qkv_f16.get_unchecked_mut(gate_index) = cvt_rn_f16_f32(qkv[gate_index]);
            }
        }
        row += row_stride;
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel uses explicit tensor and shape fields"
)]
fn xsa_backward_body(
    qkv_f16: &[u16],
    raw_out_f16: &[u16],
    mut d_out: DisjointSlice<f32>,
    mut d_qkv: DisjointSlice<f32>,
    mut d_xsa_alphas: DisjointSlice<f32>,
    alpha_bytes: &[u8],
    alpha_scales: &[u8],
    alpha_global_scale: &[f32],
    row_count: u32,
    embedding_dim: u32,
    qkv_dim: u32,
    head_count: u32,
    head_dim: u32,
    value_offset: u32,
    gate_offset: u32,
    alpha_offset: u32,
    apply_headwise_gate: u32,
    kda_value_activation: u32,
) {
    let head = thread::blockIdx_y();
    if head >= head_count {
        return;
    }
    let lane = warp::lane_id();
    let warp_in_block = thread::threadIdx_x() / 32;
    let row_stride = thread::gridDim_x() * WARPS_PER_BLOCK;
    let mut row = thread::blockIdx_x() * WARPS_PER_BLOCK + warp_in_block;
    let alpha = xsa_alpha(
        alpha_bytes,
        alpha_scales,
        alpha_global_scale,
        alpha_offset,
        head,
    );
    let mut alpha_grad_sum = 0.0;

    while row < row_count {
        let hidden_base = row * embedding_dim + head * head_dim;
        let value_base = row * qkv_dim + value_offset + head * head_dim;
        let mut value_sumsq = 0.0;
        let mut dim = lane;
        while dim < head_dim {
            let raw_value = cvt_f32_f16(qkv_f16[(value_base + dim) as usize]);
            let value = xsa_value(raw_value, kda_value_activation);
            value_sumsq += value * value;
            dim += 32;
        }
        let value_norm = max_f32(sqrt_f32(warp_sum_f32(value_sumsq)), XSA_NORM_EPS);
        let value_norm_inv = 1.0 / value_norm;

        let mut projection = 0.0;
        dim = lane;
        while dim < head_dim {
            let hidden_index = (hidden_base + dim) as usize;
            let raw_value = cvt_f32_f16(qkv_f16[(value_base + dim) as usize]);
            let value = xsa_value(raw_value, kda_value_activation) * value_norm_inv;
            projection += cvt_f32_f16(raw_out_f16[hidden_index]) * value;
            dim += 32;
        }
        let projection = warp_sum_f32(projection);
        let gate_index = (row * qkv_dim + gate_offset + head) as usize;
        let gate = if apply_headwise_gate != 0 {
            sigmoid(cvt_f32_f16(qkv_f16[gate_index]))
        } else {
            1.0
        };

        let mut gate_dot = 0.0;
        let mut grad_projection = 0.0;
        dim = lane;
        while dim < head_dim {
            let hidden_index = (hidden_base + dim) as usize;
            let raw_value = cvt_f32_f16(qkv_f16[(value_base + dim) as usize]);
            let value = xsa_value(raw_value, kda_value_activation) * value_norm_inv;
            let raw = cvt_f32_f16(raw_out_f16[hidden_index]);
            let upstream = unsafe { *d_out.as_mut_ptr().add(hidden_index) };
            let exclusive = raw - alpha * projection * value;
            gate_dot += upstream * exclusive;
            grad_projection += upstream * gate * value;
            dim += 32;
        }
        let gate_dot = warp_sum_f32(gate_dot);
        let grad_projection = warp_sum_f32(grad_projection);

        dim = lane;
        while dim < head_dim {
            let hidden_index = (hidden_base + dim) as usize;
            let value_index = (value_base + dim) as usize;
            let raw_value = cvt_f32_f16(qkv_f16[value_index]);
            let value = xsa_value(raw_value, kda_value_activation) * value_norm_inv;
            let upstream = unsafe { *d_out.as_mut_ptr().add(hidden_index) };
            let dz = upstream * gate;
            let raw = cvt_f32_f16(raw_out_f16[hidden_index]);
            let direct_value_grad = alpha
                * (-grad_projection * raw - projection * dz
                    + 2.0 * projection * grad_projection * value)
                * value_norm_inv;
            let raw_value_grad = if kda_value_activation != 0 {
                direct_value_grad * silu_grad(raw_value)
            } else {
                direct_value_grad
            };
            unsafe {
                *d_out.get_unchecked_mut(hidden_index) = dz - alpha * grad_projection * value;
                *d_qkv.get_unchecked_mut(value_index) = raw_value_grad;
            }
            dim += 32;
        }
        if lane == 0 {
            if apply_headwise_gate != 0 {
                unsafe {
                    *d_qkv.get_unchecked_mut(gate_index) = gate_dot * gate * (1.0 - gate);
                }
            }
            alpha_grad_sum += -(1.0 - alpha * alpha) * projection * grad_projection;
        }

        if head == 0 {
            let padding_start = if apply_headwise_gate != 0 {
                gate_offset + head_count
            } else {
                gate_offset
            };
            let mut padding_col = padding_start + lane;
            while padding_col < qkv_dim {
                unsafe {
                    *d_qkv.get_unchecked_mut((row * qkv_dim + padding_col) as usize) = 0.0;
                }
                padding_col += 32;
            }
        }
        row += row_stride;
    }

    if lane == 0 {
        unsafe {
            atomic_add_f32(
                d_xsa_alphas
                    .as_mut_ptr()
                    .add((alpha_offset + head) as usize),
                alpha_grad_sum,
            );
        }
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

    #[kernel]
    pub fn exclusive_self_attention_forward_kernel(
        qkv: &[f32],
        out: DisjointSlice<f32>,
        alpha_bytes: &[u8],
        alpha_scales: &[u8],
        alpha_global_scale: &[f32],
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        value_offset: u32,
        gate_offset: u32,
        alpha_offset: u32,
        apply_headwise_gate: u32,
        kda_value_activation: u32,
    ) {
        xsa_forward_body(
            qkv,
            out,
            alpha_bytes,
            alpha_scales,
            alpha_global_scale,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            value_offset,
            gate_offset,
            alpha_offset,
            apply_headwise_gate,
            kda_value_activation,
        );
    }

    #[kernel]
    pub fn exclusive_self_attention_forward_save_f16_kernel(
        qkv: &[f32],
        out: DisjointSlice<f32>,
        alpha_bytes: &[u8],
        alpha_scales: &[u8],
        alpha_global_scale: &[f32],
        qkv_f16: DisjointSlice<u16>,
        raw_out_f16: DisjointSlice<u16>,
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        value_offset: u32,
        gate_offset: u32,
        alpha_offset: u32,
        apply_headwise_gate: u32,
        kda_value_activation: u32,
    ) {
        xsa_forward_save_body(
            qkv,
            out,
            alpha_bytes,
            alpha_scales,
            alpha_global_scale,
            qkv_f16,
            raw_out_f16,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            value_offset,
            gate_offset,
            alpha_offset,
            apply_headwise_gate,
            kda_value_activation,
        );
    }

    #[kernel]
    pub fn exclusive_self_attention_backward_kernel(
        qkv_f16: &[u16],
        raw_out_f16: &[u16],
        d_out: DisjointSlice<f32>,
        d_qkv: DisjointSlice<f32>,
        d_xsa_alphas: DisjointSlice<f32>,
        alpha_bytes: &[u8],
        alpha_scales: &[u8],
        alpha_global_scale: &[f32],
        row_count: u32,
        embedding_dim: u32,
        qkv_dim: u32,
        head_count: u32,
        head_dim: u32,
        value_offset: u32,
        gate_offset: u32,
        alpha_offset: u32,
        apply_headwise_gate: u32,
        kda_value_activation: u32,
    ) {
        xsa_backward_body(
            qkv_f16,
            raw_out_f16,
            d_out,
            d_qkv,
            d_xsa_alphas,
            alpha_bytes,
            alpha_scales,
            alpha_global_scale,
            row_count,
            embedding_dim,
            qkv_dim,
            head_count,
            head_dim,
            value_offset,
            gate_offset,
            alpha_offset,
            apply_headwise_gate,
            kda_value_activation,
        );
    }
}
