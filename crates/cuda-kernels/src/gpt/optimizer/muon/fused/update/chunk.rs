use crate::amax::max4_f32;
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::max_f32;
use crate::optimizer::symexp_lin::Scalars as SymExpLinScalars;

use super::one::{UpdateAmax, update_one};

struct UpdateChunk {
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
    base: u32,
    tid: u32,
}

impl UpdateChunk {
    fn at(&self, mul: u32) -> UpdateAmax {
        update_one(
            self.u,
            self.z_master,
            self.x_master,
            self.momentum,
            self.variance,
            self.rows,
            self.cols,
            self.len,
            self.transposed,
            self.scale,
            self.polar_update_scale,
            self.normuon_factors,
            self.normuon_scale,
            self.learning_rate,
            self.weight_decay,
            self.average_coefficient,
            self.schedule_beta,
            self.symexp_lin,
            self.qk_clip_factor,
            self.base + self.tid + CTA_THREADS * mul,
        )
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_eight_amax(
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
    base: u32,
    tid: u32,
) -> UpdateAmax {
    let chunk = UpdateChunk {
        u,
        z_master,
        x_master,
        momentum,
        variance,
        rows,
        cols,
        len,
        transposed,
        scale,
        polar_update_scale,
        normuon_factors,
        normuon_scale,
        learning_rate,
        weight_decay,
        average_coefficient,
        schedule_beta,
        symexp_lin,
        qk_clip_factor,
        base,
        tid,
    };
    let v0 = chunk.at(0);
    let v1 = chunk.at(1);
    let v2 = chunk.at(2);
    let v3 = chunk.at(3);
    let v4 = chunk.at(4);
    let v5 = chunk.at(5);
    let v6 = chunk.at(6);
    let v7 = chunk.at(7);
    UpdateAmax {
        master: max_f32(
            max4_f32(v0.master, v1.master, v2.master, v3.master),
            max4_f32(v4.master, v5.master, v6.master, v7.master),
        ),
        schedule: max_f32(
            max4_f32(v0.schedule, v1.schedule, v2.schedule, v3.schedule),
            max4_f32(v4.schedule, v5.schedule, v6.schedule, v7.schedule),
        ),
    }
}
