use cuda_device::{SharedArray, thread};

use crate::block_reduce::{block_max_pair_leader_f32, block_sum_shared_f32};
use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::convert::load_f32_global_read_only;
use crate::float_ptx::{abs_f32, max_f32, sqrt_f32};

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;

const HYPERBALL_EPSILON: f32 = 1.0e-10;

pub(super) fn reduce_parameter_stats(
    chunks: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    warp_dot_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    chunk_count: u32,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp = tid / WARP_SIZE;
    let mut local_sumsq = 0.0;
    let mut local_dot = 0.0;
    let mut chunk = tid;
    while chunk < chunk_count {
        local_sumsq += read_f32(chunks, chunk);
        local_dot += read_f32(chunks, chunk_count + chunk);
        chunk += thread::blockDim_x();
    }
    let sumsq = block_sum_shared_f32(warp_sums, local_sumsq, lane, warp);
    let dot = block_sum_shared_f32(warp_dot_sums, local_dot, lane, warp);
    if tid == 0 {
        write_f32(chunks, 0, sqrt_f32(max_f32(sumsq, HYPERBALL_EPSILON)));
        write_f32(chunks, 1, dot);
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "device update uses explicit matrix state"
)]
pub(super) fn project_update_and_average_chunks(
    update: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    normuon_factors: *const f32,
    chunks: *mut f32,
    rows: u32,
    cols: u32,
    step_scale: f32,
    projection_scale: f32,
    average_coefficient: f32,
    schedule_beta: f32,
    use_schedule_free: bool,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    warp_max_pairs: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp = tid / WARP_SIZE;
    let len = rows * cols;
    let transposed = rows > cols;
    let mut local_master_amax = 0.0;
    let mut local_schedule_amax = 0.0;
    let mut index = work.thread();
    while index < len {
        let row = index / cols;
        let col = index - row * cols;
        let update_index = if transposed { col * rows + row } else { index };
        let neuron = if rows >= cols { row } else { col };
        let direction = load_f32_global_read_only(update, update_index as usize)
            * load_f32_global_read_only(normuon_factors, neuron as usize);
        let projected_z = (read_f32(z_master, index) - step_scale * direction) * projection_scale;
        let old_x = read_f32(x_master, index);
        let next_x = if use_schedule_free {
            old_x + average_coefficient * (projected_z - old_x)
        } else {
            projected_z
        };
        write_f32(z_master, index, projected_z);
        write_f32(x_master, index, next_x);
        local_master_amax = max_f32(local_master_amax, abs_f32(next_x));
        local_schedule_amax = max_f32(
            local_schedule_amax,
            abs_f32(projected_z + schedule_beta * (next_x - projected_z)),
        );
        index += work.stride();
    }
    if let Some((master_amax, schedule_amax)) = block_max_pair_leader_f32(
        warp_sums,
        warp_max_pairs,
        local_master_amax,
        local_schedule_amax,
        lane,
        warp,
    ) {
        write_f32(chunks, work.block(), master_amax);
        write_f32(chunks, work.blocks() + work.block(), schedule_amax);
    }
}
