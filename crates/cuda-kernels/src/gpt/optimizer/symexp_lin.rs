use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::atomic::atomic_add_f32;
use crate::block_reduce::block_sum_shared_f32;
use crate::float_ptx::{abs_f32, exp_f32};
use crate::warp_reduce::thread_lane_warp;

use super::threads::{APPLY_THREADS_PER_BLOCK, WARPS_PER_BLOCK};
use super::{MuonSlotDescriptor, SymExpLinSlotDescriptor};

#[inline(always)]
pub(super) fn mismatch_forward_scaled(
    raw: f32,
    beta: f32,
    exponential_scale: f32,
    linear_scale: f32,
    curvature_scale: f32,
) -> f32 {
    if beta <= 0.0 {
        return raw;
    }
    let sign = sign_f32(raw);
    if sign == 0.0 {
        return 0.0;
    }
    let exponential =
        exponential_scale * (exp_f32(beta * curvature_scale * abs_f32(raw)) - 1.0) / beta;
    sign * (exponential + linear_scale * raw / beta)
}

#[derive(Clone, Copy)]
pub(super) struct Scalars {
    beta: f32,
    exponential: f32,
    linear: f32,
    curvature: f32,
}

impl Scalars {
    #[inline(always)]
    pub(super) fn new(beta: f32, exponential: f32, linear: f32, curvature: f32) -> Self {
        Self {
            beta,
            exponential,
            linear,
            curvature,
        }
    }

    #[inline(always)]
    pub(super) fn forward(self, raw: f32) -> f32 {
        mismatch_forward_scaled(
            raw,
            self.beta,
            self.exponential,
            self.linear,
            self.curvature,
        )
    }
}

#[inline(always)]
fn mismatch_derivative(raw: f32, beta: f32) -> f32 {
    mismatch_derivative_scaled(raw, beta, 1.0, 1.0, 1.0)
}

#[inline(always)]
fn mismatch_derivative_scaled(
    raw: f32,
    beta: f32,
    exponential_scale: f32,
    linear_scale: f32,
    curvature_scale: f32,
) -> f32 {
    if beta <= 0.0 {
        return 1.0;
    }
    let sign = sign_f32(raw);
    if sign == 0.0 {
        // The paper initializes from a continuous Xavier distribution. This model's
        // NVFP4-friendly initializer contains exact zeros, so choose a live
        // subgradient rather than permanently freezing those coordinates.
        return exponential_scale * curvature_scale;
    }
    exponential_scale * curvature_scale * exp_f32(beta * curvature_scale * abs_f32(raw))
        + sign * linear_scale / beta
}

#[inline(always)]
fn congruent_inverse(target: f32, beta: f32) -> f32 {
    if beta <= 0.0 || target == 0.0 {
        return target;
    }
    let sign = sign_f32(target);
    let target_magnitude = abs_f32(target);
    let inverse_beta = 1.0 / beta;
    let mut magnitude = target_magnitude / (1.0 + inverse_beta);
    let mut iteration = 0;
    while iteration < 10 {
        let exponential = exp_f32(beta * magnitude);
        let residual = (exponential - 1.0 + magnitude) * inverse_beta - target_magnitude;
        let derivative = exponential + inverse_beta;
        magnitude -= residual / derivative;
        if magnitude < 0.0 {
            magnitude = 0.0;
        }
        iteration += 1;
    }
    sign * magnitude
}

#[inline(always)]
fn sign_f32(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn ptr_const<T>(ptr: u64) -> *const T {
    ptr as usize as *const T
}

#[inline(always)]
fn ptr_mut<T>(ptr: u64) -> *mut T {
    ptr as usize as *mut T
}

#[derive(Clone, Copy)]
pub(super) struct ScalePointers {
    pub(super) exponential_z: u64,
    pub(super) exponential_x: u64,
    pub(super) linear_z: u64,
    pub(super) linear_x: u64,
    pub(super) curvature_z: u64,
    pub(super) curvature_x: u64,
}

impl ScalePointers {
    #[inline(always)]
    pub(super) fn values(self, schedule_beta: f32) -> (f32, f32, f32) {
        if self.exponential_z == 0 {
            return (1.0, 1.0, 1.0);
        }
        unsafe {
            (
                scheduled_scalar(self.exponential_z, self.exponential_x, schedule_beta),
                scheduled_scalar(self.linear_z, self.linear_x, schedule_beta),
                scheduled_scalar(self.curvature_z, self.curvature_x, schedule_beta),
            )
        }
    }

    #[inline(always)]
    pub(super) fn scalars(self, schedule_beta: f32, beta: f32) -> Scalars {
        let (exponential, linear, curvature) = self.values(schedule_beta);
        Scalars::new(beta, exponential, linear, curvature)
    }
}

#[inline(always)]
unsafe fn scheduled_scalar(z: u64, x: u64, beta: f32) -> f32 {
    unsafe {
        let z = *ptr_const::<f32>(z);
        let x = *ptr_const::<f32>(x);
        z + beta * (x - z)
    }
}

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn symexp_lin_congruent_inverse_kernel(mut raw: DisjointSlice<f32>, beta: f32, len: u32) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            unsafe {
                let value = raw.as_mut_ptr().add(index as usize);
                *value = congruent_inverse(*value, beta);
            }
        }
    }

    #[kernel]
    pub fn symexp_lin_chain_rule_kernel(
        mut grad: DisjointSlice<f32>,
        z_master: &[f32],
        x_master: &[f32],
        schedule_beta: f32,
        beta: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            let raw = z_master[i] + schedule_beta * (x_master[i] - z_master[i]);
            unsafe {
                let value = grad.as_mut_ptr().add(i);
                *value *= mismatch_derivative(raw, beta);
            }
        }
    }

    #[kernel]
    pub fn symexp_lin_slot_chain_rule_kernel(
        slots: &[MuonSlotDescriptor],
        slot_index: u32,
        schedule_beta: f32,
        beta: f32,
    ) {
        let desc = slots[slot_index as usize];
        let len = desc.rows * desc.cols;
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            unsafe {
                let z = *ptr_const::<f32>(desc.z_master).add(i);
                let x = *ptr_const::<f32>(desc.x_master).add(i);
                let raw = z + schedule_beta * (x - z);
                let grad = ptr_mut::<f32>(desc.grad).add(i);
                *grad *= mismatch_derivative(raw, beta);
            }
        }
    }

    #[kernel]
    pub fn symexp_lin_scaled_slot_chain_rule_kernel(
        slots: &[SymExpLinSlotDescriptor],
        slot_index: u32,
        schedule_beta: f32,
        beta: f32,
    ) {
        static mut EXPONENTIAL_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut LINEAR_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut CURVATURE_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let desc = slots[slot_index as usize];
        let len = desc.rows * desc.cols;
        let (thread_id, lane, warp) = thread_lane_warp();
        let mut index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread_id;
        let stride = thread::gridDim_x() * APPLY_THREADS_PER_BLOCK;
        let scales = ScalePointers {
            exponential_z: desc.exponential_z,
            exponential_x: desc.exponential_x,
            linear_z: desc.linear_z,
            linear_x: desc.linear_x,
            curvature_z: desc.curvature_z,
            curvature_x: desc.curvature_x,
        };
        let (exponential_scale, linear_scale, curvature_scale) = scales.values(schedule_beta);
        let mut exponential_grad = 0.0;
        let mut linear_grad = 0.0;
        let mut curvature_grad = 0.0;

        while index < len {
            let i = index as usize;
            unsafe {
                let z = *ptr_const::<f32>(desc.z_master).add(i);
                let x = *ptr_const::<f32>(desc.x_master).add(i);
                let raw = z + schedule_beta * (x - z);
                let grad = ptr_mut::<f32>(desc.grad).add(i);
                let effective_grad = *grad;
                let sign = sign_f32(raw);
                let magnitude = abs_f32(raw);
                let exponential = exp_f32(beta * curvature_scale * magnitude);

                exponential_grad += effective_grad * sign * (exponential - 1.0) / beta;
                linear_grad += effective_grad * magnitude / beta;
                curvature_grad +=
                    effective_grad * sign * exponential_scale * magnitude * exponential;
                *grad = effective_grad
                    * mismatch_derivative_scaled(
                        raw,
                        beta,
                        exponential_scale,
                        linear_scale,
                        curvature_scale,
                    );
            }
            index += stride;
        }

        let (exponential_grad, linear_grad, curvature_grad) = unsafe {
            (
                block_sum_shared_f32(&mut EXPONENTIAL_SUMS, exponential_grad, lane, warp),
                block_sum_shared_f32(&mut LINEAR_SUMS, linear_grad, lane, warp),
                block_sum_shared_f32(&mut CURVATURE_SUMS, curvature_grad, lane, warp),
            )
        };
        if thread_id == 0 {
            unsafe {
                atomic_add_f32(ptr_mut::<f32>(desc.exponential_grad), exponential_grad);
                atomic_add_f32(ptr_mut::<f32>(desc.linear_grad), linear_grad);
                atomic_add_f32(ptr_mut::<f32>(desc.curvature_grad), curvature_grad);
            }
        }
    }
}
