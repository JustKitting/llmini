use cuda_device::{SharedArray, thread};

use crate::block_reduce::{block_max_pair_leader_f32, block_sum_shared_f32};
use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::convert::{
    load_bf16_global, load_f32_global_read_only, store_bf16_global,
};
use crate::float_ptx::{abs_f32, max_f32, sqrt_f32};
use crate::optimizer::MuonSlotDescriptor;
use crate::optimizer::symexp_lin::Scalars as SymExpLinScalars;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;

const HYPERBALL_EPSILON: f32 = 1.0e-10;

#[derive(Clone, Copy)]
pub(super) struct QkClip {
    factors: *const f32,
    momentum: *mut u16,
    variance: *mut u16,
    input_dim: u32,
    head_dim: u32,
    factor_offset: u32,
}

impl QkClip {
    pub(super) fn new(desc: MuonSlotDescriptor, factors: *const f32) -> Self {
        Self {
            factors,
            momentum: desc.momentum as usize as *mut u16,
            variance: desc.variance as usize as *mut u16,
            input_dim: desc.cols,
            head_dim: desc.qk_clip_head_dim,
            factor_offset: desc.qk_clip_factor_offset,
        }
    }

    #[inline(always)]
    fn apply(self, index: u32, z: f32, x: f32) -> (f32, f32) {
        if self.factor_offset == u32::MAX {
            return (z, x);
        }
        // Masters retain output-column-major Q/K/V storage in either polar
        // orientation. Clip Q/K only, after projection and before caching the
        // next training values and their amax.
        let column = index / self.input_dim;
        if column >= 2 * self.input_dim {
            return (z, x);
        }
        let head = (column % self.input_dim) / self.head_dim;
        let factor = load_f32_global_read_only(self.factors, (self.factor_offset + head) as usize);
        if factor == 1.0 {
            return (z, x);
        }
        let i = index as usize;
        store_bf16_global(
            self.momentum,
            i,
            load_bf16_global(self.momentum, i) * factor,
        );
        if !self.variance.is_null() {
            store_bf16_global(
                self.variance,
                i,
                load_bf16_global(self.variance, i) * factor * factor,
            );
        }
        (z * factor, x * factor)
    }
}
pub(super) const HYPERBALL_TRANSPOSE_TILE_ROWS: usize = 32;
pub(super) const HYPERBALL_TRANSPOSE_TILE_COLS: usize = 64;
pub(super) const HYPERBALL_TRANSPOSE_TILE_STRIDE: usize = HYPERBALL_TRANSPOSE_TILE_ROWS + 1;
pub(super) const HYPERBALL_TRANSPOSE_TILE_ELEMS: usize =
    HYPERBALL_TRANSPOSE_TILE_COLS * HYPERBALL_TRANSPOSE_TILE_STRIDE;

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
    schedule_values: *mut f32,
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
    symexp_lin: SymExpLinScalars,
    use_schedule_free: bool,
    qk_clip: QkClip,
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
        let (projected_z, next_x) = qk_clip.apply(index, projected_z, next_x);
        write_f32(z_master, index, projected_z);
        write_f32(x_master, index, next_x);
        local_master_amax = max_f32(local_master_amax, abs_f32(next_x));
        let scheduled = symexp_lin.forward(projected_z + schedule_beta * (next_x - projected_z));
        write_f32(schedule_values, index, scheduled);
        local_schedule_amax = max_f32(local_schedule_amax, abs_f32(scheduled));
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

#[expect(
    clippy::too_many_arguments,
    reason = "device update uses explicit matrix state"
)]
pub(super) fn project_transposed_update_and_average_tiles(
    update: *const f32,
    schedule_values: *mut f32,
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
    symexp_lin: SymExpLinScalars,
    use_schedule_free: bool,
    qk_clip: QkClip,
    transpose_tile: &mut SharedArray<f32, HYPERBALL_TRANSPOSE_TILE_ELEMS>,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    warp_max_pairs: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp = tid / WARP_SIZE;
    let tile_rows = HYPERBALL_TRANSPOSE_TILE_ROWS as u32;
    let tile_cols = HYPERBALL_TRANSPOSE_TILE_COLS as u32;
    let tiles_per_row = cols / tile_cols;
    let tile_row_base = (work.block() / tiles_per_row) * tile_rows;
    let tile_col_base = (work.block() % tiles_per_row) * tile_cols;

    let mut load = tid;
    while load < tile_rows * tile_cols {
        let update_row = load / tile_rows;
        let update_col = load - update_row * tile_rows;
        let update_index = (tile_col_base + update_row) * rows + tile_row_base + update_col;
        transpose_tile
            [(update_row as usize * HYPERBALL_TRANSPOSE_TILE_STRIDE) + update_col as usize] =
            load_f32_global_read_only(update, update_index as usize);
        load += thread::blockDim_x();
    }
    thread::sync_threads();

    let mut local_master_amax = 0.0;
    let mut local_schedule_amax = 0.0;
    let mut point = tid;
    while point < tile_rows * tile_cols {
        let row_in_tile = point / tile_cols;
        let col_in_tile = point - row_in_tile * tile_cols;
        let row = tile_row_base + row_in_tile;
        let col = tile_col_base + col_in_tile;
        let index = row * cols + col;
        let direction = transpose_tile
            [(col_in_tile as usize * HYPERBALL_TRANSPOSE_TILE_STRIDE) + row_in_tile as usize]
            * load_f32_global_read_only(normuon_factors, row as usize);
        let projected_z = (read_f32(z_master, index) - step_scale * direction) * projection_scale;
        let old_x = read_f32(x_master, index);
        let next_x = if use_schedule_free {
            old_x + average_coefficient * (projected_z - old_x)
        } else {
            projected_z
        };
        let (projected_z, next_x) = qk_clip.apply(index, projected_z, next_x);
        write_f32(z_master, index, projected_z);
        write_f32(x_master, index, next_x);
        local_master_amax = max_f32(local_master_amax, abs_f32(next_x));
        let scheduled = symexp_lin.forward(projected_z + schedule_beta * (next_x - projected_z));
        write_f32(schedule_values, index, scheduled);
        local_schedule_amax = max_f32(local_schedule_amax, abs_f32(scheduled));
        point += thread::blockDim_x();
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
