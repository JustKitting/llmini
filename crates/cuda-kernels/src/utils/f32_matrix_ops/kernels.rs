use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::{block_max_store_f32, block_sum_shared_f32};
use crate::f16_tc_matmul::convert::{load_f32x2_global, store_f32x2_global};
use crate::float_ptx::{abs_f32, fma_f32, max_f32, sqrt_f32};
use crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK;
use crate::warp_reduce::thread_lane_warp;

const F32_OPS_WARPS_PER_BLOCK: usize = 8;

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn f32_linear2_kernel(
        a: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        len: u32,
        a_scale: f32,
        b_scale: f32,
    ) {
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let i = index as usize;
            unsafe {
                *out.get_unchecked_mut(i) = fma_f32(a_scale, a[i], b_scale * b[i]);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_linear3_in_place_kernel(
        a: &[f32],
        b: &[f32],
        mut c_out: DisjointSlice<f32>,
        len: u32,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let c_ptr = c_out.as_mut_ptr();
        while index < len {
            let i = index as usize;
            unsafe {
                let current = *c_ptr.add(i);
                let bc = fma_f32(b_scale, b[i], c_scale * current);
                *c_ptr.add(i) = fma_f32(a_scale, a[i], bc);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_linear3_sqrt_bound_a_in_place_kernel(
        a: &[f32],
        b: &[f32],
        mut c_out: DisjointSlice<f32>,
        bound_amax: &[f32],
        len: u32,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        let bound = bound_amax[0];
        let bound_scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        let c_ptr = c_out.as_mut_ptr();
        while index < len {
            let i = index as usize;
            unsafe {
                let current = *c_ptr.add(i);
                let bc = fma_f32(b_scale, b[i], c_scale * current);
                *c_ptr.add(i) = fma_f32(a_scale, a[i] * bound_scale, bc);
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_linear3_sqrt_bound_a_amax_in_place_kernel(
        a: &[f32],
        b: &[f32],
        mut c_out: DisjointSlice<f32>,
        bound_amax: &[f32],
        mut chunk_amax: DisjointSlice<f32>,
        len: u32,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        static mut CHUNK_AMAX: SharedArray<f32, F32_OPS_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let bound = bound_amax[0];
        let bound_scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let chunk = thread::blockIdx_x();
        let (thread, lane, warp) = thread_lane_warp();
        let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
        let mut offset = thread * 2;
        let mut local_amax = 0.0;
        let c_ptr = c_out.as_mut_ptr();
        while offset < TENSOR_AMAX_VALUES_PER_BLOCK {
            let index = base + offset;
            if index + 1 < len {
                let i = index as usize;
                let (a0, a1) = load_f32x2_global(a.as_ptr(), i);
                let (b0, b1) = load_f32x2_global(b.as_ptr(), i);
                let (c0, c1) = load_f32x2_global(c_ptr, i);
                let value0 = fma_f32(
                    a_scale,
                    a0 * bound_scale,
                    fma_f32(b_scale, b0, c_scale * c0),
                );
                let value1 = fma_f32(
                    a_scale,
                    a1 * bound_scale,
                    fma_f32(b_scale, b1, c_scale * c1),
                );
                store_f32x2_global(c_ptr, i, value0, value1);
                local_amax = max_f32(local_amax, abs_f32(value0));
                local_amax = max_f32(local_amax, abs_f32(value1));
            } else if index < len {
                let i = index as usize;
                unsafe {
                    let current = *c_ptr.add(i);
                    let bc = fma_f32(b_scale, b[i], c_scale * current);
                    let value = fma_f32(a_scale, a[i] * bound_scale, bc);
                    *c_ptr.add(i) = value;
                    local_amax = max_f32(local_amax, abs_f32(value));
                }
            }
            offset += thread::blockDim_x() * 2;
        }
        block_max_store_f32!(CHUNK_AMAX, chunk_amax[chunk], local_amax, lane, warp);
    }

    #[kernel]
    pub fn f32_linear3_sqrt_bound_a_row_sumsq_in_place_kernel(
        a: &[f32],
        b: &[f32],
        mut c_out: DisjointSlice<f32>,
        bound_amax: &[f32],
        mut row_sumsq: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        static mut ROW_SUMS: SharedArray<f32, F32_OPS_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let bound = bound_amax[0];
        let bound_scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let (thread, lane, warp) = thread_lane_warp();
        let row_base = row as usize * cols as usize;
        let c_ptr = c_out.as_mut_ptr();
        let mut local_sumsq = 0.0;
        let mut col = thread * 2;
        while col < cols {
            let i = row_base + col as usize;
            if col + 1 < cols {
                let (a0, a1) = load_f32x2_global(a.as_ptr(), i);
                let (b0, b1) = load_f32x2_global(b.as_ptr(), i);
                let (c0, c1) = load_f32x2_global(c_ptr, i);
                let value0 = fma_f32(
                    a_scale,
                    a0 * bound_scale,
                    fma_f32(b_scale, b0, c_scale * c0),
                );
                let value1 = fma_f32(
                    a_scale,
                    a1 * bound_scale,
                    fma_f32(b_scale, b1, c_scale * c1),
                );
                store_f32x2_global(c_ptr, i, value0, value1);
                local_sumsq = fma_f32(value0, value0, local_sumsq);
                local_sumsq = fma_f32(value1, value1, local_sumsq);
            } else {
                unsafe {
                    let current = *c_ptr.add(i);
                    let bc = fma_f32(b_scale, b[i], c_scale * current);
                    let value = fma_f32(a_scale, a[i] * bound_scale, bc);
                    *c_ptr.add(i) = value;
                    local_sumsq = fma_f32(value, value, local_sumsq);
                }
            }
            col += thread::blockDim_x() * 2;
        }
        let sumsq = unsafe { block_sum_shared_f32(&mut ROW_SUMS, local_sumsq, lane, warp) };
        if thread == 0 {
            unsafe {
                *row_sumsq.get_unchecked_mut(row as usize) = sumsq;
            }
        }
    }

    #[kernel]
    pub fn f32_add_scaled_identity_kernel(
        src: &[f32],
        mut out: DisjointSlice<f32>,
        dim: u32,
        scale: f32,
    ) {
        let len = dim * dim;
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let add = if index.is_multiple_of(dim + 1) {
                scale
            } else {
                0.0
            };
            unsafe {
                *out.get_unchecked_mut(index as usize) = src[index as usize] + add;
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_scale_in_place_by_sqrt_amax_bound_kernel(
        mut x: DisjointSlice<f32>,
        amax: &[f32],
        len: u32,
    ) {
        let bound = amax[0];
        let scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let i = index as usize;
            unsafe {
                *x.get_unchecked_mut(i) *= scale;
            }
            index += stride;
        }
    }

    #[kernel]
    pub fn f32_scale_in_place_by_amax_bound_kernel(
        mut x: DisjointSlice<f32>,
        amax: &[f32],
        len: u32,
    ) {
        let bound = amax[0];
        let scale = if bound > 1.0 { 1.0 / bound } else { 1.0 };
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        while index < len {
            let i = index as usize;
            unsafe {
                *x.get_unchecked_mut(i) *= scale;
            }
            index += stride;
        }
    }
}
