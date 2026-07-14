use crate::amax::max4_f32;
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::max_f32;

use super::one::update_one;

struct UpdateChunk {
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    base: u32,
    tid: u32,
}

impl UpdateChunk {
    fn at(&self, mul: u32) -> f32 {
        update_one(
            self.u,
            self.z_master,
            self.x_master,
            self.rows,
            self.cols,
            self.len,
            self.transposed,
            self.scale,
            self.learning_rate,
            self.weight_decay,
            self.average_coefficient,
            self.base + self.tid + CTA_THREADS * mul,
        )
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_eight_amax(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    base: u32,
    tid: u32,
) -> f32 {
    let chunk = UpdateChunk {
        u,
        z_master,
        x_master,
        rows,
        cols,
        len,
        transposed,
        scale,
        learning_rate,
        weight_decay,
        average_coefficient,
        base,
        tid,
    };
    max_f32(
        max4_f32(chunk.at(0), chunk.at(1), chunk.at(2), chunk.at(3)),
        max4_f32(chunk.at(4), chunk.at(5), chunk.at(6), chunk.at(7)),
    )
}
