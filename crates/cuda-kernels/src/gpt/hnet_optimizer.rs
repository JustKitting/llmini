use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DeviceCopy, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::{
    f16_tc_matmul::convert::cvt_f32_bf16,
    float_ptx::sqrt_f32,
    launch::linear_config,
    nvfp4::{Nvfp4DeviceTensor, nvfp4_value},
};

const THREADS: u32 = 256;

pub struct CompactAdamModule(kernels::LoadedModule);

pub struct CompactAdamState {
    z: DeviceBuffer<u16>,
    x: DeviceBuffer<u16>,
    first: DeviceBuffer<u16>,
    second: DeviceBuffer<u16>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompactAdamStep {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub beta1_correction: f32,
    pub beta2_correction: f32,
    pub eps: f32,
    pub average: f32,
    pub grad_scale: f32,
    pub seed: u32,
}

unsafe impl DeviceCopy for CompactAdamStep {}

impl CompactAdamModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self(kernels::from_module(module)?))
    }

    pub fn from_nvfp4(
        &self,
        stream: &CudaStream,
        weight: Nvfp4DeviceTensor<'_>,
        len: u32,
    ) -> Result<CompactAdamState, DriverError> {
        if len == 0 || !len.is_multiple_of(16) || weight.bytes.len() < len as usize / 2 {
            return Err(DriverError(1));
        }
        let mut state = CompactAdamState {
            z: DeviceBuffer::zeroed(stream, len as usize)?,
            x: DeviceBuffer::zeroed(stream, len as usize)?,
            first: DeviceBuffer::zeroed(stream, len as usize)?,
            second: DeviceBuffer::zeroed(stream, len as usize)?,
        };
        self.0.hnet_adam_initialize_kernel(
            stream,
            linear_config(len, THREADS),
            weight.bytes,
            weight.scales,
            weight.global_scale,
            &mut state.z,
            &mut state.x,
            len,
        )?;
        Ok(state)
    }

    pub fn update(
        &self,
        stream: &CudaStream,
        state: &mut CompactAdamState,
        gradient: &DeviceBuffer<f32>,
        step: CompactAdamStep,
    ) -> Result<(), DriverError> {
        if state.z.len() != gradient.len() {
            return Err(DriverError(1));
        }
        let len = state.z.len() as u32;
        self.0.hnet_adam_update_kernel(
            stream,
            linear_config(len, THREADS),
            &mut state.z,
            &mut state.x,
            &mut state.first,
            &mut state.second,
            gradient,
            step,
            len,
        )
    }

    /// Write the scheduled view into shared materialization scratch. Beta = 1
    /// selects the averaged parameters for holdout scoring.
    pub fn write_view(
        &self,
        stream: &CudaStream,
        state: &CompactAdamState,
        beta: f32,
        scratch: &mut DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if scratch.len() < state.z.len() {
            return Err(DriverError(1));
        }
        let len = state.z.len() as u32;
        self.0.hnet_adam_write_view_kernel(
            stream,
            linear_config(len, THREADS),
            &state.z,
            &state.x,
            scratch,
            beta,
            len,
        )
    }
}

#[inline(always)]
fn stochastic_bfloat16(value: f32, random: u32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        // Truncation alone can turn an FP32 NaN with a short payload into Inf.
        return ((bits >> 16) | if bits & 0x007f_ffff == 0 { 0 } else { 0x40 }) as u16;
    }
    let rounded = ((bits + (random & 0xffff)) >> 16) as u16;
    if rounded & 0x7fff == 0x7f80 {
        rounded - 1
    } else {
        rounded
    }
}

#[inline(always)]
fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn hnet_adam_initialize_kernel(
        bytes: &[u8],
        scales: &[u8],
        global: &[f32],
        mut z: DisjointSlice<u16>,
        mut x: DisjointSlice<u16>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < len {
            let at = index as usize;
            let value = nvfp4_value(bytes, scales, global[0], at).to_bits();
            let initial = ((value + 0x7fff + ((value >> 16) & 1)) >> 16) as u16;
            unsafe {
                *z.get_unchecked_mut(at) = initial;
                *x.get_unchecked_mut(at) = initial;
            }
        }
    }

    #[kernel]
    pub fn hnet_adam_update_kernel(
        mut z: DisjointSlice<u16>,
        mut x: DisjointSlice<u16>,
        mut first: DisjointSlice<u16>,
        mut second: DisjointSlice<u16>,
        gradient: &[f32],
        step: CompactAdamStep,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < len {
            let at = index as usize;
            let g = gradient[at] * step.grad_scale;
            unsafe {
                let z = z.get_unchecked_mut(at);
                let x = x.get_unchecked_mut(at);
                let first = first.get_unchecked_mut(at);
                let second = second.get_unchecked_mut(at);
                let m = step.beta1 * cvt_f32_bf16(*first) + (1.0 - step.beta1) * g;
                let v = step.beta2 * cvt_f32_bf16(*second) + (1.0 - step.beta2) * g * g;
                let direction =
                    (m / step.beta1_correction) / (sqrt_f32(v / step.beta2_correction) + step.eps);
                let next = cvt_f32_bf16(*z) * (1.0 - step.learning_rate * step.weight_decay)
                    - step.learning_rate * direction;
                let average = cvt_f32_bf16(*x);
                let next_average = average + step.average * (next - average);
                let seed = hash(step.seed ^ index);
                *first = stochastic_bfloat16(m, hash(seed ^ 0x7432_9da3));
                *second = stochastic_bfloat16(v, hash(seed ^ 0x58c5_182d));
                *z = stochastic_bfloat16(next, hash(seed ^ 0xe1d7_0c2f));
                *x = stochastic_bfloat16(next_average, hash(seed ^ 0x47b6_09af));
            }
        }
    }

    #[kernel]
    pub fn hnet_adam_write_view_kernel(
        z: &[u16],
        x: &[u16],
        mut scratch: DisjointSlice<f32>,
        beta: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < len {
            let at = index as usize;
            let next = cvt_f32_bf16(z[at]);
            let average = cvt_f32_bf16(x[at]);
            let value = if beta == 1.0 {
                average
            } else {
                next + beta * (average - next)
            };
            unsafe { *scratch.get_unchecked_mut(at) = value };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stochastic_bfloat16;

    #[test]
    fn hnet_stochastic_rounding_keeps_the_expected_low_precision_state() {
        for bits in [0x3f80_2357, 0xbf80_0123, 0x2467_c7a8] {
            let truncated = (bits >> 16) as u16;
            let up = (0..=0xffff)
                .filter(|&random| {
                    stochastic_bfloat16(f32::from_bits(bits), random) == truncated + 1
                })
                .count();
            assert_eq!(up, (bits & 0xffff) as usize);
        }
        assert_eq!(stochastic_bfloat16(0.0, u32::MAX), 0);
        assert_eq!(stochastic_bfloat16(f32::from_bits(0x7f80_0001), 0), 0x7fc0);
    }
}
