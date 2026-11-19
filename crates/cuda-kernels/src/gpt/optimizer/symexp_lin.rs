use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use crate::float_ptx::{abs_f32, exp_f32, log1p_f32, sqrt_f32};

use super::SymExpLinSlotDescriptor;
use super::threads::APPLY_THREADS_PER_BLOCK;

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
fn congruent_inverse(target: f32, beta: f32) -> f32 {
    if beta <= 0.0 || target == 0.0 {
        return target;
    }
    let sign = sign_f32(target);
    let target_magnitude = abs_f32(target);
    let inverse_beta = 1.0 / beta;
    // A tight upper bound: omitting the positive linear term gives
    // log1p(beta * target)/beta. The linear approximation can start so far
    // above large trained weights that ten Newton steps never converge.
    let mut magnitude = log1p_f32(beta * target_magnitude) * inverse_beta;
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

#[expect(
    clippy::too_many_arguments,
    reason = "device helper mirrors the explicit AdamW state"
)]
#[inline(always)]
unsafe fn scalar_adamw_update(
    z_master_ptr: u64,
    x_master_ptr: u64,
    grad_ptr: u64,
    first_ptr: u64,
    second_ptr: u64,
    grad_scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    eps: f32,
    average_coefficient: f32,
) {
    unsafe {
        let z_master = ptr_mut::<f32>(z_master_ptr);
        let x_master = ptr_mut::<f32>(x_master_ptr);
        let first = ptr_mut::<f32>(first_ptr);
        let second = ptr_mut::<f32>(second_ptr);
        let g = *ptr_const::<f32>(grad_ptr) * grad_scale;
        let m = beta1 * *first + (1.0 - beta1) * g;
        let v = beta2 * *second + (1.0 - beta2) * g * g;
        let update = (m / beta1_correction) / (sqrt_f32(v / beta2_correction) + eps);
        let current = *z_master;
        let decay = 1.0 - learning_rate * weight_decay;
        let next = current * decay - learning_rate * update;
        let x = *x_master;

        *first = m;
        *second = v;
        *z_master = next;
        *x_master = x + average_coefficient * (next - x);
    }
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
    pub fn symexp_lin_zero_scale_grads_kernel(slots: &[SymExpLinSlotDescriptor], slot_count: u32) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < slot_count {
            let desc = slots[index as usize];
            if desc.exponential_grad != 0 {
                unsafe {
                    *ptr_mut::<f32>(desc.exponential_grad) = 0.0;
                    *ptr_mut::<f32>(desc.linear_grad) = 0.0;
                    *ptr_mut::<f32>(desc.curvature_grad) = 0.0;
                }
            }
        }
    }

    #[kernel]
    pub fn symexp_lin_scale_adamw_update_kernel(
        slots: &[SymExpLinSlotDescriptor],
        slot_count: u32,
        grad_scale: f32,
        exponential_linear_learning_rate: f32,
        curvature_learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < slot_count {
            let desc = slots[index as usize];
            if desc.exponential_z != 0 {
                unsafe {
                    scalar_adamw_update(
                        desc.exponential_z,
                        desc.exponential_x,
                        desc.exponential_grad,
                        desc.exponential_first,
                        desc.exponential_second,
                        grad_scale,
                        exponential_linear_learning_rate,
                        weight_decay,
                        beta1,
                        beta2,
                        beta1_correction,
                        beta2_correction,
                        eps,
                        average_coefficient,
                    );
                    scalar_adamw_update(
                        desc.linear_z,
                        desc.linear_x,
                        desc.linear_grad,
                        desc.linear_first,
                        desc.linear_second,
                        grad_scale,
                        exponential_linear_learning_rate,
                        weight_decay,
                        beta1,
                        beta2,
                        beta1_correction,
                        beta2_correction,
                        eps,
                        average_coefficient,
                    );
                    scalar_adamw_update(
                        desc.curvature_z,
                        desc.curvature_x,
                        desc.curvature_grad,
                        desc.curvature_first,
                        desc.curvature_second,
                        grad_scale,
                        curvature_learning_rate,
                        0.0,
                        beta1,
                        beta2,
                        beta1_correction,
                        beta2_correction,
                        eps,
                        average_coefficient,
                    );
                }
            }
        }
    }

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
}
