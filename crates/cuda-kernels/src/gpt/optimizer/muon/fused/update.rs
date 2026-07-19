use cuda_device::{SharedArray, thread};

use crate::amax::max4_f32;
use crate::f16_tc_matmul::convert::load_f32_global_read_only;
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::max_f32;
use crate::float_ptx::sqrt_f32;

use super::super::super::symexp_lin::Scalars as SymExpLinScalars;
use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;

mod chunk;
mod one;
use one::update_one_sign;

const UPDATE_VALUES_PER_CHUNK: u32 = crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_master_chunks(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
    variance: *mut u16,
    block_amax: *mut f32,
    normuon_factors: *const f32,
    normuon_scale: f32,
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
    symexp_lin: SymExpLinScalars,
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
            load_f32_global_read_only(qk_clip_factors, (qk_clip_factor_offset + head) as usize)
        } else {
            1.0
        };
        let local_amax = chunk::update_eight_amax(
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

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_sign_master_chunks(
    grad: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    momentum: *mut f32,
    variance: *mut u16,
    block_amax: *mut f32,
    qk_clip_factors: *const f32,
    qk_clip_factor_offset: u32,
    qk_clip_head_dim: u32,
    rows: u32,
    len: u32,
    mu: f32,
    grad_scale: f32,
    variance_adaptive: bool,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    symexp_lin: SymExpLinScalars,
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

    while chunk < chunk_count {
        let base = chunk * UPDATE_VALUES_PER_CHUNK;
        let qk_clip_factor = if qk_clip_factor_offset != u32::MAX && chunk < 2 * rows {
            debug_assert_eq!(rows, UPDATE_VALUES_PER_CHUNK);
            debug_assert!(qk_clip_head_dim > 0);
            let head = (chunk % rows) / qk_clip_head_dim;
            load_f32_global_read_only(qk_clip_factors, (qk_clip_factor_offset + head) as usize)
        } else {
            1.0
        };
        let v0 = update_one_sign(
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
            base + tid,
        );
        let v1 = update_one_sign(
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
            base + tid + CTA_THREADS,
        );
        let v2 = update_one_sign(
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
            base + tid + 2 * CTA_THREADS,
        );
        let v3 = update_one_sign(
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
            base + tid + 3 * CTA_THREADS,
        );
        let v4 = update_one_sign(
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
            base + tid + 4 * CTA_THREADS,
        );
        let v5 = update_one_sign(
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
            base + tid + 5 * CTA_THREADS,
        );
        let v6 = update_one_sign(
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
            base + tid + 6 * CTA_THREADS,
        );
        let v7 = update_one_sign(
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
            base + tid + 7 * CTA_THREADS,
        );
        local_master_amax = max_f32(
            local_master_amax,
            max_f32(
                max4_f32(v0.master, v1.master, v2.master, v3.master),
                max4_f32(v4.master, v5.master, v6.master, v7.master),
            ),
        );
        local_schedule_amax = max_f32(
            local_schedule_amax,
            max_f32(
                max4_f32(v0.schedule, v1.schedule, v2.schedule, v3.schedule),
                max4_f32(v4.schedule, v5.schedule, v6.schedule, v7.schedule),
            ),
        );
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
