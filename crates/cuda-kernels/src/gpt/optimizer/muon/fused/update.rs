use cuda_device::{SharedArray, thread};

use crate::float_ptx::max_f32;
use crate::float_ptx::sqrt_f32;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;

mod chunk;
mod one;

const UPDATE_VALUES_PER_CHUNK: u32 = crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_master_chunks(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
    block_amax: *mut f32,
    qk_clip_factors: *const f32,
    qk_clip_factor_offset: u32,
    qk_clip_head_dim: u32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    polar_update_scale: f32,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    warp_max_pairs: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp_in_block = tid / WARP_SIZE;
    let mut chunk = work.block();
    let chunk_count = len.div_ceil(UPDATE_VALUES_PER_CHUNK);
    let mut local_master_amax = 0.0;
    let mut local_schedule_amax = 0.0;
    let scale = 0.2 * sqrt_f32(max_f32(rows as f32, cols as f32));

    while chunk < chunk_count {
        let base = chunk * UPDATE_VALUES_PER_CHUNK;
        let qk_clip_factor = if qk_clip_factor_offset != u32::MAX && chunk < 2 * rows {
            debug_assert_eq!(rows, UPDATE_VALUES_PER_CHUNK);
            debug_assert!(qk_clip_head_dim > 0);
            let head = (chunk % rows) / qk_clip_head_dim;
            unsafe { *qk_clip_factors.add((qk_clip_factor_offset + head) as usize) }
        } else {
            1.0
        };
        let local_amax = chunk::update_eight_amax(
            u,
            z_master,
            x_master,
            momentum,
            rows,
            cols,
            len,
            transposed,
            scale,
            polar_update_scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            schedule_beta,
            qk_clip_factor,
            base,
            tid,
        );
        local_master_amax = max_f32(local_master_amax, local_amax.master);
        local_schedule_amax = max_f32(local_schedule_amax, local_amax.schedule);
        chunk += work.blocks();
    }
    if let Some((master_amax, schedule_amax)) = crate::block_reduce::block_max_pair_leader_f32(
        warp_sums,
        warp_max_pairs,
        local_master_amax,
        local_schedule_amax,
        lane,
        warp_in_block,
    ) {
        unsafe {
            *block_amax.add(work.block() as usize) = master_amax;
            *block_amax.add((work.blocks() + work.block()) as usize) = schedule_amax;
        }
    }
}
