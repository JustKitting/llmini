use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::{
    block_reduce::block_sum_shared_f32,
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::{fma_f32, sqrt_f32},
    launch::{grid_x_config, linear_config},
    warp_reduce::{thread_lane_warp, warp_sum_f32},
};

const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;

pub struct FrameModule(kernels::LoadedModule);

impl FrameModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self(kernels::from_module(module)?))
    }

    pub fn load_warm(
        &self,
        stream: &CudaStream,
        warm: &DeviceBuffer<u16>,
        frame: &mut DeviceBuffer<f32>,
        seed: u32,
        width: u32,
        cold: bool,
    ) -> Result<(), DriverError> {
        let len = warm.len() as u32;
        self.0.echelon_load_frame_kernel(
            stream,
            linear_config(len, THREADS),
            warm,
            frame,
            len,
            seed,
            width,
            cold as u32,
        )
    }

    pub fn store_warm(
        &self,
        stream: &CudaStream,
        frame: &DeviceBuffer<f32>,
        warm: &mut DeviceBuffer<u16>,
    ) -> Result<(), DriverError> {
        let len = warm.len() as u32;
        self.0
            .echelon_store_frame_kernel(stream, linear_config(len, THREADS), frame, warm, len)
    }

    /// Turn a batch of transposed thin factors into transposed orthonormal
    /// frames. The factors become reflector storage during the operation.
    pub fn orthonormalize(
        &self,
        stream: &CudaStream,
        factors: &mut DeviceBuffer<f32>,
        frames: &mut DeviceBuffer<f32>,
        batch: u32,
        rank: u32,
        vector_len: u32,
    ) -> Result<(), DriverError> {
        let elements = rank
            .checked_mul(vector_len)
            .and_then(|size| size.checked_mul(batch))
            .filter(|&size| size != 0 && rank <= vector_len)
            .ok_or(DriverError(1))? as usize;
        if factors.len() < elements || frames.len() < elements {
            return Err(DriverError(1));
        }
        self.0.echelon_householder_frame_kernel(
            stream,
            grid_x_config(batch, THREADS),
            factors,
            frames,
            rank,
            vector_len,
        )
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn echelon_load_frame_kernel(
        warm: &[u16],
        mut frame: DisjointSlice<f32>,
        len: u32,
        seed: u32,
        width: u32,
        cold: u32,
    ) {
        let index = thread::blockIdx_x() * THREADS + thread::threadIdx_x();
        if index < len {
            let value = if cold == 0 {
                cvt_f32_f16(warm[index as usize])
            } else {
                super::super::signed_unit(index, seed, width)
            };
            unsafe { *frame.get_unchecked_mut(index as usize) = value };
        }
    }

    #[kernel]
    pub fn echelon_store_frame_kernel(frame: &[f32], mut warm: DisjointSlice<u16>, len: u32) {
        let index = (thread::blockIdx_x() * THREADS + thread::threadIdx_x()) as usize;
        if index < len as usize {
            unsafe { *warm.get_unchecked_mut(index) = cvt_rn_f16_f32(frame[index]) };
        }
    }

    /// Apply one reflector to independent vectors. Each warp owns a target
    /// vector, and the caller synchronizes the block between reflectors.
    #[inline(always)]
    unsafe fn apply_reflector(
        reflector: *const f32,
        target: *mut f32,
        first: u32,
        len: u32,
        lane: u32,
        tau: f32,
    ) {
        let v = |col: u32| {
            if col == first {
                1.0
            } else {
                unsafe { *reflector.add(col as usize) }
            }
        };
        let mut dot = 0.0;
        for col in (first + lane..len).step_by(32) {
            let value = unsafe { *target.add(col as usize) };
            dot = fma_f32(v(col), value, dot);
        }
        let update = tau * warp_sum_f32(dot);
        for col in (first + lane..len).step_by(32) {
            let element = unsafe { &mut *target.add(col as usize) };
            *element = fma_f32(-v(col), update, *element);
        }
    }

    #[kernel]
    pub fn echelon_householder_frame_kernel(
        mut factors: DisjointSlice<f32>,
        mut frames: DisjointSlice<f32>,
        rank: u32,
        vector_len: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        static mut REFLECTOR: SharedArray<f32, 2> = SharedArray::UNINIT;
        let (tid, lane, warp) = thread_lane_warp();
        let elements = rank * vector_len;
        let base = (thread::blockIdx_x() * elements) as usize;
        let factors = unsafe { factors.as_mut_ptr().add(base) };
        let frames = unsafe { frames.as_mut_ptr().add(base) };
        // Keep one implicit leading 1 and the reflector coefficient for
        // each vector. This operates on the thin matrix directly.
        for k in 0..rank {
            let reflector = unsafe { factors.add((k * vector_len) as usize) };
            let mut tail_square = 0.0;
            for col in (k + 1 + tid..vector_len).step_by(THREADS as usize) {
                let value = unsafe { *reflector.add(col as usize) };
                tail_square = fma_f32(value, value, tail_square);
            }
            let tail_square =
                unsafe { block_sum_shared_f32(&mut REDUCTION, tail_square, lane, warp) };
            if tid == 0 {
                let diagonal = unsafe { *reflector.add(k as usize) };
                let (tau, denominator) = if tail_square == 0.0 {
                    (0.0, 1.0)
                } else {
                    let norm = sqrt_f32(fma_f32(diagonal, diagonal, tail_square));
                    let alpha = if diagonal >= 0.0 { -norm } else { norm };
                    (1.0 - diagonal / alpha, diagonal - alpha)
                };
                unsafe {
                    REFLECTOR[0] = tau;
                    REFLECTOR[1] = denominator;
                }
            }
            thread::sync_threads();
            let (tau, denominator) = unsafe { (REFLECTOR[0], REFLECTOR[1]) };
            if tid == 0 {
                unsafe { *reflector.add(k as usize) = tau };
            }
            if tau != 0.0 {
                for col in (k + 1 + tid..vector_len).step_by(THREADS as usize) {
                    unsafe { *reflector.add(col as usize) /= denominator };
                }
            }
            thread::sync_threads();
            if tau != 0.0 {
                for j in (k + 1 + warp..rank).step_by(WARPS) {
                    let column = unsafe { factors.add((j * vector_len) as usize) };
                    unsafe { apply_reflector(reflector, column, k, vector_len, lane, tau) };
                }
            }
            thread::sync_threads();
        }
        for at in (tid..elements).step_by(THREADS as usize) {
            let value = if at / vector_len == at % vector_len {
                1.0
            } else {
                0.0
            };
            unsafe { *frames.add(at as usize) = value };
        }
        thread::sync_threads();
        for k in (0..rank).rev() {
            let reflector = unsafe { factors.add((k * vector_len) as usize) };
            let tau = unsafe { *reflector.add(k as usize) };
            if tau != 0.0 {
                for j in (k + warp..rank).step_by(WARPS) {
                    let column = unsafe { frames.add((j * vector_len) as usize) };
                    unsafe { apply_reflector(reflector, column, k, vector_len, lane, tau) };
                }
            }
            thread::sync_threads();
        }
    }
}
