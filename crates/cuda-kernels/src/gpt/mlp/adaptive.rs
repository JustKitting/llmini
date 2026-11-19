use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::{MlpModule, kernels::BLOCK_TOPK_TILE};
use crate::{
    block_reduce::block_sum_shared_f32,
    f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32},
    float_ptx::max_f32,
    launch::grid_x_config,
    warp_reduce::thread_lane_warp,
};

const THREADS: u32 = 256;
const WARPS: usize = (THREADS / 32) as usize;

impl MlpModule {
    /// Replace the raw affine output by ReLU² minus its per-token feature mean.
    /// Reverse uses the ordinary FP16 preactivation tape.
    pub fn adaptive_relu2_forward(
        &self,
        stream: &CudaStream,
        values: &mut DeviceBuffer<f32>,
        tape: Option<&mut DeviceBuffer<u16>>,
        rows: u32,
        width: u32,
    ) -> Result<(), DriverError> {
        let len = rows
            .checked_mul(width)
            .filter(|&n| n != 0)
            .ok_or(DriverError(1))? as usize;
        if values.len() < len || tape.as_ref().is_some_and(|saved| saved.len() < len) {
            return Err(DriverError(1));
        }
        let tape = tape.map_or(core::ptr::null_mut(), |saved| {
            saved.cu_deviceptr() as *mut u16
        });
        self.adaptive
            .as_ref()
            .ok_or(DriverError(1))?
            .adaptive_relu2_forward_kernel(
                stream,
                grid_x_config(rows, THREADS),
                values,
                tape,
                width,
            )
    }

    /// Apply the coupled centering adjoint in place to the down-projection
    /// input gradient. A route masks only the direct term of that gradient.
    pub fn adaptive_relu2_backward(
        &self,
        stream: &CudaStream,
        gradient: &mut DeviceBuffer<f32>,
        tape: &DeviceBuffer<u16>,
        masks: &DeviceBuffer<u64>,
        rows: u32,
        width: u32,
        routed: bool,
    ) -> Result<(), DriverError> {
        let len = rows
            .checked_mul(width)
            .filter(|&n| n != 0)
            .ok_or(DriverError(1))? as usize;
        if gradient.len() < len
            || tape.len() < len
            || (routed
                && (width > 64 * BLOCK_TOPK_TILE
                    || masks.len() < rows.div_ceil(BLOCK_TOPK_TILE) as usize))
        {
            return Err(DriverError(1));
        }
        self.adaptive
            .as_ref()
            .ok_or(DriverError(1))?
            .adaptive_relu2_backward_kernel(
                stream,
                grid_x_config(rows, THREADS),
                gradient,
                tape,
                masks,
                width,
                routed as u32,
            )
    }
}

#[cuda_module]
pub(super) mod kernels {
    use super::*;

    #[kernel]
    pub fn adaptive_relu2_forward_kernel(
        mut values: DisjointSlice<f32>,
        tape: *mut u16,
        width: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let base = (thread::blockIdx_x() * width) as usize;
        let (tid, lane, warp) = thread_lane_warp();
        let row = unsafe { values.as_mut_ptr().add(base) };
        let mut sum = 0.0;
        for col in (tid..width).step_by(THREADS as usize) {
            let z = unsafe { *row.add(col as usize) };
            if !tape.is_null() {
                unsafe { *tape.add(base + col as usize) = cvt_rn_f16_f32(z) };
            }
            let positive = max_f32(z, 0.0);
            sum += positive * positive;
        }
        let mean = unsafe { block_sum_shared_f32(&mut REDUCTION, sum, lane, warp) } / width as f32;
        for col in (tid..width).step_by(THREADS as usize) {
            let value = unsafe { &mut *row.add(col as usize) };
            let positive = max_f32(*value, 0.0);
            *value = positive * positive - mean;
        }
    }

    #[kernel]
    pub fn adaptive_relu2_backward_kernel(
        mut gradient: DisjointSlice<f32>,
        tape: &[u16],
        masks: &[u64],
        width: u32,
        routed: u32,
    ) {
        static mut REDUCTION: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let row = thread::blockIdx_x();
        let base = (row * width) as usize;
        let (tid, lane, warp) = thread_lane_warp();
        let mask = if routed != 0 {
            masks[(row / BLOCK_TOPK_TILE) as usize]
        } else {
            u64::MAX
        };
        let gradient = unsafe { gradient.as_mut_ptr().add(base) };
        let direct = |col: u32| {
            if mask & (1_u64 << ((col / BLOCK_TOPK_TILE) & 63)) != 0 {
                unsafe { *gradient.add(col as usize) }
            } else {
                0.0
            }
        };
        let mut sum = 0.0;
        for col in (tid..width).step_by(THREADS as usize) {
            sum += direct(col);
        }
        let mean = unsafe { block_sum_shared_f32(&mut REDUCTION, sum, lane, warp) } / width as f32;
        for col in (tid..width).step_by(THREADS as usize) {
            let positive = max_f32(cvt_f32_f16(tape[base + col as usize]), 0.0);
            let adjoint = (direct(col) - mean) * 2.0 * positive;
            unsafe { *gradient.add(col as usize) = adjoint };
        }
    }
}
