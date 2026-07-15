use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::{fma_f32, sqrt_f32};

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
