//! Causal RMS input of latent feedback. Dense products use shared NVFP4 TMA.
use crate::{block_reduce::block_sum_shared_f32, float_ptx::sqrt_f32};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
#[cuda_module]
pub mod kernels {
    use super::*;
    static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
    #[kernel]
    pub fn latent_feedback_shifted_rms_kernel(
        top: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        d: u32,
        seq: u32,
        epsilon: f32,
    ) {
        let row = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        if row >= n {
            return;
        }
        if row % seq == 0 {
            let mut j = tid;
            while j < d {
                unsafe {
                    *out.get_unchecked_mut((row * d + j) as usize) = 0.0;
                }
                j += 256;
            }
            return;
        }
        let base = (row - 1) * d;
        let mut square = 0.0;
        let mut j = tid;
        while j < d {
            let x = top[(base + j) as usize];
            square += x * x;
            j += 256;
        }
        let square = unsafe { block_sum_shared_f32(&mut REDUCE, square, tid % 32, tid / 32) };
        let inverse = 1.0 / sqrt_f32(square / d as f32 + epsilon);
        j = tid;
        while j < d {
            unsafe {
                *out.get_unchecked_mut((row * d + j) as usize) = top[(base + j) as usize] * inverse;
            }
            j += 256;
        }
    }
}
