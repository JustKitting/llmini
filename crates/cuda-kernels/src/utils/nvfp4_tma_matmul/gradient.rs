use crate::atomic::atomic_add_f32;
use crate::f16_tc_matmul::convert::load_f32x2_global_read_only;
use crate::float_ptx::{abs_f32, exp_f32};
use crate::optimizer::SymExpLinSlotDescriptor;

/// Parameters for transforming a matrix gradient while its GEMM accumulators
/// are still in registers. All pointers refer to the existing optimizer state.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GemmGradientTransform {
    pub z_master: u64,
    pub x_master: u64,
    pub exponential_z: u64,
    pub exponential_x: u64,
    pub exponential_grad: u64,
    pub linear_z: u64,
    pub linear_x: u64,
    pub linear_grad: u64,
    pub curvature_z: u64,
    pub curvature_x: u64,
    pub curvature_grad: u64,
    pub norm_sums: u64,
    pub schedule_beta: f32,
    pub beta: f32,
    pub learn_scales: u32,
    pub len: u32,
}

impl GemmGradientTransform {
    pub fn new(
        desc: SymExpLinSlotDescriptor,
        norm_sums: u64,
        schedule_beta: f32,
        beta: f32,
        learn_scales: bool,
    ) -> Self {
        Self {
            z_master: desc.z_master,
            x_master: desc.x_master,
            exponential_z: desc.exponential_z,
            exponential_x: desc.exponential_x,
            exponential_grad: desc.exponential_grad,
            linear_z: desc.linear_z,
            linear_x: desc.linear_x,
            linear_grad: desc.linear_grad,
            curvature_z: desc.curvature_z,
            curvature_x: desc.curvature_x,
            curvature_grad: desc.curvature_grad,
            norm_sums,
            schedule_beta,
            beta,
            learn_scales: learn_scales as u32,
            len: desc.rows * desc.cols,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GradientSums {
    pub exponential: f32,
    pub linear: f32,
    pub curvature: f32,
    pub norm: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct GradientScalars {
    params: GemmGradientTransform,
    exponential: f32,
    curvature: f32,
    inverse_beta: f32,
    exponential_derivative: f32,
    linear_derivative: f32,
}

impl GradientScalars {
    #[inline(always)]
    pub fn new(params: GemmGradientTransform) -> Self {
        let (exponential, linear, curvature) =
            if params.learn_scales != 0 && params.exponential_z != 0 {
                unsafe {
                    (
                        scheduled(
                            params.exponential_z,
                            params.exponential_x,
                            params.schedule_beta,
                        ),
                        scheduled(params.linear_z, params.linear_x, params.schedule_beta),
                        scheduled(params.curvature_z, params.curvature_x, params.schedule_beta),
                    )
                }
            } else {
                (1.0, 1.0, 1.0)
            };
        let inverse_beta = 1.0 / params.beta;
        Self {
            params,
            exponential,
            curvature,
            inverse_beta,
            exponential_derivative: exponential * curvature,
            linear_derivative: linear * inverse_beta,
        }
    }

    #[inline(always)]
    fn value(self, raw: f32, grad: f32, sums: &mut GradientSums) -> f32 {
        let sign = if raw > 0.0 {
            1.0
        } else if raw < 0.0 {
            -1.0
        } else {
            0.0
        };
        let magnitude = abs_f32(raw);
        let exponential = exp_f32(self.params.beta * self.curvature * magnitude);
        if self.params.learn_scales != 0 {
            sums.exponential += grad * sign * (exponential - 1.0) * self.inverse_beta;
            sums.linear += grad * magnitude * self.inverse_beta;
            sums.curvature += grad * sign * self.exponential * magnitude * exponential;
        }
        let transformed =
            grad * (self.exponential_derivative * exponential + sign * self.linear_derivative);
        sums.norm += transformed * transformed;
        transformed
    }

    #[inline(always)]
    pub fn pair(self, index: u32, grad0: f32, grad1: f32, sums: &mut GradientSums) -> (f32, f32) {
        if index >= self.params.len {
            return (grad0, grad1);
        }
        let (z0, z1) = load_f32x2_global_read_only(
            self.params.z_master as usize as *const f32,
            index as usize,
        );
        let (x0, x1) = load_f32x2_global_read_only(
            self.params.x_master as usize as *const f32,
            index as usize,
        );
        (
            self.value(z0 + self.params.schedule_beta * (x0 - z0), grad0, sums),
            self.value(z1 + self.params.schedule_beta * (x1 - z1), grad1, sums),
        )
    }
}

#[inline(always)]
unsafe fn scheduled(z: u64, x: u64, beta: f32) -> f32 {
    unsafe {
        let z = *(z as usize as *const f32);
        let x = *(x as usize as *const f32);
        z + beta * (x - z)
    }
}

#[inline(always)]
pub(crate) unsafe fn store_sums(
    params: GemmGradientTransform,
    cta: u32,
    exponential: f32,
    linear: f32,
    curvature: f32,
    norm: f32,
) {
    if params.learn_scales != 0 {
        unsafe {
            atomic_add_f32(params.exponential_grad as usize as *mut f32, exponential);
            atomic_add_f32(params.linear_grad as usize as *mut f32, linear);
            atomic_add_f32(params.curvature_grad as usize as *mut f32, curvature);
            *(params.norm_sums as usize as *mut f32).add(cta as usize) = norm;
        }
    }
}
