use crate::float_ptx::abs_f32;

use crate::f16_tc_matmul::convert::{
    load_bf16_global, load_f32_global_read_only, store_bf16_global,
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
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
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
            *momentum.add(index as usize) *= qk_clip_factor;
            if !variance.is_null() {
                let variance_value =
                    load_bf16_global(variance, index as usize) * qk_clip_factor * qk_clip_factor;
                store_bf16_global(variance, index as usize, variance_value);
            }
        }
        *z = next_z;
        *x = next_x;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(symexp_lin.forward(next_z + schedule_beta * (next_x - next_z))),
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_one_sign(
    grad: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
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
        let momentum_ptr = momentum.add(index as usize);
        let g = *grad.add(index as usize) * grad_scale;
        let previous_momentum = *momentum_ptr;
        let next_momentum = mu * previous_momentum + (1.0 - mu) * g;
        let mut next_variance = 0.0;
        let direction_source = if variance_adaptive {
            let innovation = previous_momentum - g;
            next_variance = mu * load_bf16_global(variance, index as usize)
                + mu * (1.0 - mu) * innovation * innovation;
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
        *momentum_ptr = clipped_momentum;
        *z = next_z;
        *x = next_x;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(symexp_lin.forward(next_z + schedule_beta * (next_x - next_z))),
        }
    }
}
