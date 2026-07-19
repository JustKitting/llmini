use super::super::super::work_grid::WorkGrid;
use super::types::MuonMatrixShape;
use crate::device_ptr::write_f32;
use crate::f16_tc_matmul::convert::{
    load_bf16_global, load_bf16x2_global, load_f32x2_global, load_f32x2_global_read_only,
    store_bf16_global, store_bf16x2_global, store_f32x2_global,
};
use crate::float_ptx::{max_f32, sqrt_f32};

const MUON_VS_EPSILON: f32 = 1.0e-8;

pub(super) fn momentum_orient(
    grad: *const f32,
    momentum: *mut f32,
    variance: *mut u16,
    oriented: *mut f32,
    work: WorkGrid,
    shape: MuonMatrixShape,
    mu: f32,
    grad_scale: f32,
    transposed: bool,
    nesterov: bool,
    variance_adaptive: bool,
    bias_correction_inv: f32,
) {
    if variance_adaptive {
        momentum_orient_variance_adaptive(
            grad,
            momentum,
            variance,
            oriented,
            work,
            shape,
            mu,
            grad_scale,
            transposed,
            nesterov,
            bias_correction_inv,
        );
        return;
    }

    let len = shape.len();
    let mut index = work.thread();
    while index < len {
        let row = index / shape.cols;
        let col = index - row * shape.cols;
        let g = unsafe { *grad.add(index as usize) } * grad_scale;
        unsafe {
            let momentum_ptr = momentum.add(index as usize);
            let previous_momentum = *momentum_ptr;
            let next_momentum = mu * previous_momentum + (1.0 - mu) * g;
            let update = if nesterov {
                mu * next_momentum + (1.0 - mu) * g
            } else {
                next_momentum
            };
            *momentum_ptr = next_momentum;
            let dst = if transposed {
                col * shape.rows + row
            } else {
                index
            };
            write_f32(oriented, dst, update);
        }
        index += work.stride();
    }
}

#[expect(clippy::too_many_arguments, reason = "fused CUDA path")]
fn momentum_orient_variance_adaptive(
    grad: *const f32,
    momentum: *mut f32,
    variance: *mut u16,
    oriented: *mut f32,
    work: WorkGrid,
    shape: MuonMatrixShape,
    mu: f32,
    grad_scale: f32,
    transposed: bool,
    nesterov: bool,
    bias_correction_inv: f32,
) {
    let len = shape.len();
    let one_minus_mu = 1.0 - mu;
    let variance_factor = mu * one_minus_mu;
    let nesterov_factor = mu / one_minus_mu;
    let mut index = work.thread() * 2;
    let stride = work.stride() * 2;

    while index + 1 < len {
        let i = index as usize;
        let (raw_g0, raw_g1) = load_f32x2_global_read_only(grad, i);
        let (previous_momentum0, previous_momentum1) = load_f32x2_global(momentum, i);
        let (previous_variance0, previous_variance1) = load_bf16x2_global(variance, i);
        let g0 = raw_g0 * grad_scale;
        let g1 = raw_g1 * grad_scale;
        let next_momentum0 = mu * previous_momentum0 + one_minus_mu * g0;
        let next_momentum1 = mu * previous_momentum1 + one_minus_mu * g1;
        let innovation0 = previous_momentum0 - g0;
        let innovation1 = previous_momentum1 - g1;
        let next_variance0 = mu * previous_variance0 + variance_factor * innovation0 * innovation0;
        let next_variance1 = mu * previous_variance1 + variance_factor * innovation1 * innovation1;
        let corrected_momentum0 = next_momentum0 * bias_correction_inv;
        let corrected_momentum1 = next_momentum1 * bias_correction_inv;
        let numerator0 = if nesterov {
            g0 + nesterov_factor * corrected_momentum0
        } else {
            corrected_momentum0
        };
        let numerator1 = if nesterov {
            g1 + nesterov_factor * corrected_momentum1
        } else {
            corrected_momentum1
        };
        let update0 = numerator0
            / (sqrt_f32(max_f32(next_variance0 * bias_correction_inv, 0.0)) + MUON_VS_EPSILON);
        let update1 = numerator1
            / (sqrt_f32(max_f32(next_variance1 * bias_correction_inv, 0.0)) + MUON_VS_EPSILON);

        store_f32x2_global(momentum, i, next_momentum0, next_momentum1);
        store_bf16x2_global(variance, i, next_variance0, next_variance1);
        if transposed {
            let row0 = index / shape.cols;
            let col0 = index - row0 * shape.cols;
            let index1 = index + 1;
            let row1 = index1 / shape.cols;
            let col1 = index1 - row1 * shape.cols;
            write_f32(oriented, col0 * shape.rows + row0, update0);
            write_f32(oriented, col1 * shape.rows + row1, update1);
        } else {
            store_f32x2_global(oriented, i, update0, update1);
        }
        index += stride;
    }

    if index < len {
        momentum_orient_variance_adaptive_tail(
            grad,
            momentum,
            variance,
            oriented,
            shape,
            mu,
            grad_scale,
            transposed,
            nesterov,
            bias_correction_inv,
            index,
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "fused CUDA tail path")]
fn momentum_orient_variance_adaptive_tail(
    grad: *const f32,
    momentum: *mut f32,
    variance: *mut u16,
    oriented: *mut f32,
    shape: MuonMatrixShape,
    mu: f32,
    grad_scale: f32,
    transposed: bool,
    nesterov: bool,
    bias_correction_inv: f32,
    index: u32,
) {
    let row = index / shape.cols;
    let col = index - row * shape.cols;
    unsafe {
        let g = *grad.add(index as usize) * grad_scale;
        let momentum_ptr = momentum.add(index as usize);
        let previous_momentum = *momentum_ptr;
        let next_momentum = mu * previous_momentum + (1.0 - mu) * g;
        let innovation = previous_momentum - g;
        let next_variance = mu * load_bf16_global(variance, index as usize)
            + mu * (1.0 - mu) * innovation * innovation;
        let corrected_momentum = next_momentum * bias_correction_inv;
        let numerator = if nesterov {
            g + (mu / (1.0 - mu)) * corrected_momentum
        } else {
            corrected_momentum
        };
        let update = numerator
            / (sqrt_f32(max_f32(next_variance * bias_correction_inv, 0.0)) + MUON_VS_EPSILON);
        *momentum_ptr = next_momentum;
        store_bf16_global(variance, index as usize, next_variance);
        let dst = if transposed {
            col * shape.rows + row
        } else {
            index
        };
        write_f32(oriented, dst, update);
    }
}
