use crate::float_ptx::abs_f32;

use crate::f16_tc_matmul::convert::{
    load_bf16_global, load_bf16x2_global, load_f32_global_read_only, load_f32x2_global,
    load_f32x2_global_read_only, store_bf16_global, store_bf16x2_global, store_f32x2_global,
};
use crate::optimizer::symexp_lin::Scalars as SymExpLinScalars;

#[derive(Clone, Copy)]
pub(super) struct UpdateAmax {
    pub(super) master: f32,
    pub(super) schedule: f32,
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_one(
    u: *const f32,
    schedule_values: *mut f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut u16,
    variance: *mut u16,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    scale: f32,
    polar_update_scale: f32,
    normuon_factors: *const f32,
    normuon_scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    symexp_lin: SymExpLinScalars,
    qk_clip_factor: f32,
    index: u32,
) -> UpdateAmax {
    if index >= len {
        return UpdateAmax {
            master: 0.0,
            schedule: 0.0,
        };
    }

    let row = index / cols;
    let col = index - row * cols;
    let update_index = if transposed { col * rows + row } else { index };
    let normuon_factor = if normuon_factors.is_null() {
        1.0
    } else {
        let neuron = if rows >= cols { row } else { col };
        load_f32_global_read_only(normuon_factors, neuron as usize) * normuon_scale
    };
    let muon_update = scale
        * (load_f32_global_read_only(u, update_index as usize)
            * polar_update_scale
            * normuon_factor);
    let decay = 1.0 - learning_rate * weight_decay;

    unsafe {
        let z = z_master.add(index as usize);
        let x = x_master.add(index as usize);
        let mut next_z = *z * decay - learning_rate * muon_update;
        let mut next_x = *x + average_coefficient * (next_z - *x);
        if qk_clip_factor != 1.0 {
            next_z *= qk_clip_factor;
            next_x *= qk_clip_factor;
            let clipped = load_bf16_global(momentum, index as usize) * qk_clip_factor;
            store_bf16_global(momentum, index as usize, clipped);
            if !variance.is_null() {
                let variance_value =
                    load_bf16_global(variance, index as usize) * qk_clip_factor * qk_clip_factor;
                store_bf16_global(variance, index as usize, variance_value);
            }
        }
        *z = next_z;
        *x = next_x;
        let scheduled = symexp_lin.forward(next_z + schedule_beta * (next_x - next_z));
        *schedule_values.add(index as usize) = scheduled;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(scheduled),
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_one_sign(
    grad: *mut f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut u16,
    variance: *mut u16,
    len: u32,
    mu: f32,
    grad_scale: f32,
    variance_adaptive: bool,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    symexp_lin: SymExpLinScalars,
    qk_clip_factor: f32,
    index: u32,
) -> UpdateAmax {
    if index >= len {
        return UpdateAmax {
            master: 0.0,
            schedule: 0.0,
        };
    }

    unsafe {
        let g = *grad.add(index as usize) * grad_scale;
        let previous_momentum = load_bf16_global(momentum, index as usize);
        let one_minus_mu = 1.0 - mu;
        let next_momentum = mu * previous_momentum + one_minus_mu * g;
        let mut next_variance = 0.0;
        let direction_source = if variance_adaptive {
            let innovation = previous_momentum - g;
            next_variance = mu * load_bf16_global(variance, index as usize)
                + mu * one_minus_mu * innovation * innovation;
            next_momentum
        } else {
            next_momentum
        };
        let direction = if direction_source > 0.0 {
            1.0
        } else if direction_source < 0.0 {
            -1.0
        } else {
            0.0
        };
        let decay = 1.0 - learning_rate * weight_decay;
        let z = z_master.add(index as usize);
        let x = x_master.add(index as usize);
        let mut next_z = *z * decay - learning_rate * direction;
        let mut next_x = *x + average_coefficient * (next_z - *x);
        let mut clipped_momentum = next_momentum;
        if qk_clip_factor != 1.0 {
            next_z *= qk_clip_factor;
            next_x *= qk_clip_factor;
            clipped_momentum *= qk_clip_factor;
            if variance_adaptive {
                next_variance *= qk_clip_factor * qk_clip_factor;
            }
        }
        if variance_adaptive {
            store_bf16_global(variance, index as usize, next_variance);
        }
        store_bf16_global(momentum, index as usize, clipped_momentum);
        *z = next_z;
        *x = next_x;
        let scheduled = symexp_lin.forward(next_z + schedule_beta * (next_x - next_z));
        *grad.add(index as usize) = scheduled;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(scheduled),
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_sign_pair(
    grad: *mut f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut u16,
    variance: *mut u16,
    len: u32,
    mu: f32,
    grad_scale: f32,
    variance_adaptive: bool,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    symexp_lin: SymExpLinScalars,
    qk_clip_factor: f32,
    index: u32,
) -> (UpdateAmax, UpdateAmax) {
    if index + 1 >= len {
        return (
            update_one_sign(
                grad,
                z_master,
                x_master,
                momentum,
                variance,
                len,
                mu,
                grad_scale,
                variance_adaptive,
                learning_rate,
                weight_decay,
                average_coefficient,
                schedule_beta,
                symexp_lin,
                qk_clip_factor,
                index,
            ),
            update_one_sign(
                grad,
                z_master,
                x_master,
                momentum,
                variance,
                len,
                mu,
                grad_scale,
                variance_adaptive,
                learning_rate,
                weight_decay,
                average_coefficient,
                schedule_beta,
                symexp_lin,
                qk_clip_factor,
                index + 1,
            ),
        );
    }

    let i = index as usize;
    let (raw_g0, raw_g1) = load_f32x2_global_read_only(grad as *const f32, i);
    let (previous_momentum0, previous_momentum1) = load_bf16x2_global(momentum, i);
    let g0 = raw_g0 * grad_scale;
    let g1 = raw_g1 * grad_scale;
    let one_minus_mu = 1.0 - mu;
    let next_momentum0 = mu * previous_momentum0 + one_minus_mu * g0;
    let next_momentum1 = mu * previous_momentum1 + one_minus_mu * g1;
    let mut next_variance0 = 0.0;
    let mut next_variance1 = 0.0;
    if variance_adaptive {
        let (previous_variance0, previous_variance1) = load_bf16x2_global(variance, i);
        let variance_factor = mu * one_minus_mu;
        let innovation0 = previous_momentum0 - g0;
        let innovation1 = previous_momentum1 - g1;
        next_variance0 = mu * previous_variance0 + variance_factor * innovation0 * innovation0;
        next_variance1 = mu * previous_variance1 + variance_factor * innovation1 * innovation1;
    }
    let direction0 = if next_momentum0 > 0.0 {
        1.0
    } else if next_momentum0 < 0.0 {
        -1.0
    } else {
        0.0
    };
    let direction1 = if next_momentum1 > 0.0 {
        1.0
    } else if next_momentum1 < 0.0 {
        -1.0
    } else {
        0.0
    };
    let mut clipped_momentum0 = next_momentum0;
    let mut clipped_momentum1 = next_momentum1;
    if qk_clip_factor != 1.0 {
        clipped_momentum0 *= qk_clip_factor;
        clipped_momentum1 *= qk_clip_factor;
        if variance_adaptive {
            let variance_clip = qk_clip_factor * qk_clip_factor;
            next_variance0 *= variance_clip;
            next_variance1 *= variance_clip;
        }
    }
    if variance_adaptive {
        store_bf16x2_global(variance, i, next_variance0, next_variance1);
    }
    store_bf16x2_global(momentum, i, clipped_momentum0, clipped_momentum1);

    let decay = 1.0 - learning_rate * weight_decay;
    let (z0, z1) = load_f32x2_global(z_master, i);
    let (x0, x1) = load_f32x2_global(x_master, i);
    let mut next_z0 = z0 * decay - learning_rate * direction0;
    let mut next_z1 = z1 * decay - learning_rate * direction1;
    let mut next_x0 = x0 + average_coefficient * (next_z0 - x0);
    let mut next_x1 = x1 + average_coefficient * (next_z1 - x1);
    if qk_clip_factor != 1.0 {
        next_z0 *= qk_clip_factor;
        next_z1 *= qk_clip_factor;
        next_x0 *= qk_clip_factor;
        next_x1 *= qk_clip_factor;
    }
    store_f32x2_global(z_master, i, next_z0, next_z1);
    store_f32x2_global(x_master, i, next_x0, next_x1);
    let scheduled0 = symexp_lin.forward(next_z0 + schedule_beta * (next_x0 - next_z0));
    let scheduled1 = symexp_lin.forward(next_z1 + schedule_beta * (next_x1 - next_z1));
    store_f32x2_global(grad, i, scheduled0, scheduled1);
    (
        UpdateAmax {
            master: abs_f32(next_x0),
            schedule: abs_f32(scheduled0),
        },
        UpdateAmax {
            master: abs_f32(next_x1),
            schedule: abs_f32(scheduled1),
        },
    )
}
