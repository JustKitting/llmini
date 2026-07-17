use crate::float_ptx::abs_f32;

use crate::f16_tc_matmul::convert::load_f32_global_read_only;

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
        }
        *z = next_z;
        *x = next_x;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(next_z + schedule_beta * (next_x - next_z)),
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_one_sign(
    grad: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
    len: u32,
    mu: f32,
    grad_scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
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
        let next_momentum = mu * *momentum_ptr + (1.0 - mu) * g;
        let direction = if next_momentum > 0.0 {
            1.0
        } else if next_momentum < 0.0 {
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
        }
        *momentum_ptr = clipped_momentum;
        *z = next_z;
        *x = next_x;
        UpdateAmax {
            master: abs_f32(next_x),
            schedule: abs_f32(next_z + schedule_beta * (next_x - next_z)),
        }
    }
}
