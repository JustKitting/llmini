//! The original finite20-round mHC coefficient program distributed over a warp.
//! Keep sequential four-term sums/divisions and the exact raw adjoint. Only
//! independent matrix cells/tokens move to different lanes; no dense projection,
//! parameter, optimizer, quantization, or RNG equation changes.
use crate::{
    atomic::atomic_add_f32,
    float_ptx::{exp_f32, max_f32},
};
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};
use std::sync::Arc;
const EPS: f32 = 1e-6;
pub struct MhcCoefficientsModule {
    kernels: kernels::LoadedModule,
}
pub struct ForwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub raw: &'a DeviceBuffer<f32>,
    pub bias: &'a DeviceBuffer<f32>,
    pub alpha: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub stride: u32,
    pub head: bool,
}
pub struct BackwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub raw: &'a DeviceBuffer<f32>,
    pub bias: &'a DeviceBuffer<f32>,
    pub alpha: &'a DeviceBuffer<f32>,
    pub dc: &'a DeviceBuffer<f32>,
    pub draw: &'out mut DeviceBuffer<f32>,
    pub dbias: &'out mut DeviceBuffer<f32>,
    pub dalpha: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub stride: u32,
    pub head: bool,
}
fn shape(n: u32, stride: u32, head: bool) -> Result<(usize, usize, usize, usize), DriverError> {
    let count = if head { 4 } else { 24 };
    let groups = if head { 1 } else { 3 };
    if n == 0 || stride < count {
        return Err(DriverError(1));
    }
    let raw = n.checked_mul(stride).ok_or(DriverError(1))? as usize;
    let coeff = n.checked_mul(24).ok_or(DriverError(1))? as usize;
    Ok((raw, coeff, count as usize, groups))
}
fn config(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(8), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
impl MhcCoefficientsModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        for name in [
            b"mhc_coefficients_warp_forward_kernel".as_slice(),
            b"mhc_coefficients_warp_backward_kernel".as_slice(),
        ] {
            if !crate::module_exports::contains(&module, name)? {
                return Err(DriverError(1));
            }
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn forward(&self, a: ForwardArgs<'_, '_>) -> Result<(), DriverError> {
        let (raw, coeff, count, groups) = shape(a.rows, a.stride, a.head)?;
        if a.raw.len() < raw
            || a.out.len() < coeff
            || a.bias.len() < count
            || a.alpha.len() < groups
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_coefficients_warp_forward_kernel(
            a.stream,
            config(a.rows),
            a.raw,
            a.bias,
            a.alpha,
            a.out,
            a.rows,
            a.stride,
            a.head,
        )
    }
    pub fn backward(&self, a: BackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let (raw, coeff, count, groups) = shape(a.rows, a.stride, a.head)?;
        if a.raw.len() < raw
            || a.draw.len() < raw
            || a.dc.len() < coeff
            || a.bias.len() < count
            || a.dbias.len() < count
            || a.alpha.len() < groups
            || a.dalpha.len() < groups
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_coefficients_warp_backward_kernel(
            a.stream,
            config(a.rows),
            a.raw,
            a.bias,
            a.alpha,
            a.dc,
            a.draw,
            a.dbias,
            a.dalpha,
            a.rows,
            a.stride,
            a.head,
        )
    }
}
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}
// Both halves of a warp hold the same16 cells. All32 lanes execute every
// shuffle with the full mask; only the first copy is written. Thus tail tokens
// can return by whole warp, with no partial-warp collective or CTA barrier.
#[inline(always)]
fn initial(raw: &[f32], bias: &[f32], alpha: f32, start: usize, lane: u32) -> f32 {
    let cell = (lane & 15) as usize;
    let first = (cell / 4 * 4) as u32;
    let z = raw[start + 8 + cell] * alpha + bias[8 + cell];
    let mut maximum = f32::NEG_INFINITY;
    let mut j = 0;
    while j < 4 {
        maximum = max_f32(maximum, warp::shuffle_f32(z, first + j));
        j += 1;
    }
    let value = exp_f32(z - maximum);
    let mut total = 0.0;
    j = 0;
    while j < 4 {
        total += warp::shuffle_f32(value, first + j);
        j += 1;
    }
    value / total + EPS
}
#[inline(always)]
fn source(lane: u32, stage: usize, j: u32) -> u32 {
    let cell = lane & 15;
    if stage % 2 == 1 {
        cell % 4 + j * 4
    } else {
        cell / 4 * 4 + j
    }
}
#[inline(always)]
fn normalize(value: f32, lane: u32, stage: usize) -> f32 {
    let mut total = 0.0;
    let mut j = 0;
    while j < 4 {
        total += warp::shuffle_f32(value, source(lane, stage, j));
        j += 1;
    }
    value / (total + EPS)
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn mhc_coefficients_warp_forward_kernel(
        raw: &[f32],
        bias: &[f32],
        alpha: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        stride: u32,
        head: bool,
    ) {
        let tid = thread::threadIdx_x();
        let row = thread::blockIdx_x() * 8 + tid / 32;
        let lane = tid & 31;
        if row >= n {
            return;
        }
        let start = (row * stride) as usize;
        let b = (row * 24) as usize;
        if lane < 4 {
            unsafe {
                *out.get_unchecked_mut(b + lane as usize) =
                    sigmoid(raw[start + lane as usize] * alpha[0] + bias[lane as usize]) + EPS;
            }
        }
        if head {
            if lane >= 4 && lane < 24 {
                unsafe {
                    *out.get_unchecked_mut(b + lane as usize) = 0.0;
                }
            }
            return;
        }
        if lane >= 4 && lane < 8 {
            unsafe {
                *out.get_unchecked_mut(b + lane as usize) =
                    2.0 * sigmoid(raw[start + lane as usize] * alpha[1] + bias[lane as usize]);
            }
        }
        let mut matrix = initial(raw, bias, alpha[2], start, lane);
        let mut stage = 1;
        while stage < 40 {
            matrix = normalize(matrix, lane, stage);
            stage += 1;
        }
        if lane < 16 {
            unsafe {
                *out.get_unchecked_mut(b + 8 + lane as usize) = matrix;
            }
        }
    }
    #[kernel]
    pub fn mhc_coefficients_warp_backward_kernel(
        raw: &[f32],
        bias: &[f32],
        alpha: &[f32],
        dc: &[f32],
        mut draw: DisjointSlice<f32>,
        mut dbias: DisjointSlice<f32>,
        mut dalpha: DisjointSlice<f32>,
        n: u32,
        stride: u32,
        head: bool,
    ) {
        let tid = thread::threadIdx_x();
        let row = thread::blockIdx_x() * 8 + tid / 32;
        let lane = tid & 31;
        if row >= n {
            return;
        }
        let start = (row * stride) as usize;
        let b = (row * 24) as usize;
        let mut matrix_derivative = 0.0;
        if !head {
            let mut history = [0.0f32; 40];
            let mut matrix = initial(raw, bias, alpha[2], start, lane);
            history[0] = matrix;
            let mut stage = 1;
            while stage < 40 {
                matrix = normalize(matrix, lane, stage);
                history[stage] = matrix;
                stage += 1;
            }
            let mut g = dc[b + 8 + (lane & 15) as usize];
            stage = 39;
            while stage > 0 {
                let previous = unsafe { *history.get_unchecked(stage - 1) };
                let current = unsafe { *history.get_unchecked(stage) };
                let mut denom = 0.0;
                let mut dot = 0.0;
                let mut j = 0;
                while j < 4 {
                    let at = source(lane, stage, j);
                    denom += warp::shuffle_f32(previous, at);
                    dot += warp::shuffle_f32(g, at) * warp::shuffle_f32(current, at);
                    j += 1;
                }
                g = (g - dot) / (denom + EPS);
                stage -= 1;
            }
            let probability = history[0] - EPS;
            let first = (lane & 15) / 4 * 4;
            let mut dot = 0.0;
            let mut j = 0;
            while j < 4 {
                dot += warp::shuffle_f32(g, first + j) * warp::shuffle_f32(probability, first + j);
                j += 1;
            }
            matrix_derivative = probability * (g - dot);
        }
        let residual = warp::shuffle_f32(matrix_derivative, (lane + 8) & 15);
        let count = if head { 4 } else { 24 };
        let groups = if head { 1 } else { 3 };
        let derivative = if lane < 4 {
            let s = sigmoid(raw[start + lane as usize] * alpha[0] + bias[lane as usize]);
            dc[b + lane as usize] * s * (1.0 - s)
        } else if !head && lane < 8 {
            let s = sigmoid(raw[start + lane as usize] * alpha[1] + bias[lane as usize]);
            dc[b + lane as usize] * 2.0 * s * (1.0 - s)
        } else if !head && lane < 24 {
            residual
        } else {
            0.0
        };
        let raw_value = if lane < count {
            raw[start + lane as usize]
        } else {
            0.0
        };
        let mut i = lane;
        while i < stride {
            unsafe {
                *draw.get_unchecked_mut(start + i as usize) = 0.0;
            }
            i += 32;
        }
        if lane < count {
            let group = if lane < 4 {
                0
            } else if lane < 8 {
                1
            } else {
                2
            };
            unsafe {
                *draw.get_unchecked_mut(start + lane as usize) = derivative * alpha[group];
                atomic_add_f32(dbias.as_mut_ptr().add(lane as usize), derivative);
            }
        }
        let mut group = 0;
        while group < groups {
            let first = if group == 0 {
                0
            } else if group == 1 {
                4
            } else {
                8
            };
            let end = if group == 0 {
                4
            } else if group == 1 {
                8
            } else {
                24
            };
            let mut total = 0.0;
            i = first;
            while i < end {
                total += warp::shuffle_f32(derivative, i) * warp::shuffle_f32(raw_value, i);
                i += 1;
            }
            if lane == group {
                unsafe {
                    atomic_add_f32(dalpha.as_mut_ptr().add(group as usize), total);
                }
            }
            group += 1;
        }
    }
}
