use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::block_sum_shared_f32;
use crate::float_ptx::{max_f32, sqrt_f32};
use crate::warp_reduce::thread_lane_warp;

pub const EMBER_COLUMN_ROW_TILE: u32 = 128;
pub const EMBER_THREADS_PER_BLOCK: u32 = 256;
const EMBER_WARPS_PER_BLOCK: usize = (EMBER_THREADS_PER_BLOCK / 32) as usize;

pub const fn ember_column_partial_len(rows: u32, cols: u32) -> usize {
    rows.div_ceil(EMBER_COLUMN_ROW_TILE) as usize * cols as usize
}

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn ember_row_second_moment_kernel(
        grad: &[f32],
        grad_scale: f32,
        mut row_second_moment: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        beta2: f32,
    ) {
        static mut REDUCE: SharedArray<f32, EMBER_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let (thread, lane, warp) = thread_lane_warp();
        let row_base = row * cols;
        let mut sum = 0.0;
        let mut col = thread;
        while col < cols {
            let value = grad[(row_base + col) as usize] * grad_scale;
            sum += value * value;
            col += EMBER_THREADS_PER_BLOCK;
        }
        let sum = unsafe { block_sum_shared_f32(&mut REDUCE, sum, lane, warp) };
        if thread == 0 {
            let index = row as usize;
            unsafe {
                let old = *row_second_moment.as_mut_ptr().add(index);
                *row_second_moment.as_mut_ptr().add(index) =
                    beta2 * old + (1.0 - beta2) * (sum / cols as f32);
            }
        }
    }

    #[kernel]
    pub fn ember_column_partial_kernel(
        grad: &[f32],
        grad_scale: f32,
        mut column_partials: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let column_tiles = cols.div_ceil(EMBER_THREADS_PER_BLOCK);
        let tile = thread::blockIdx_x();
        let row_tile = tile / column_tiles;
        let column_tile = tile - row_tile * column_tiles;
        let col = column_tile * EMBER_THREADS_PER_BLOCK + thread::threadIdx_x();
        if col >= cols {
            return;
        }

        let row_start = row_tile * EMBER_COLUMN_ROW_TILE;
        let row_end = {
            let end = row_start + EMBER_COLUMN_ROW_TILE;
            if end < rows { end } else { rows }
        };
        let mut sum = 0.0;
        let mut row = row_start;
        while row < row_end {
            let value = grad[(row * cols + col) as usize] * grad_scale;
            sum += value * value;
            row += 1;
        }
        unsafe {
            *column_partials
                .as_mut_ptr()
                .add((row_tile * cols + col) as usize) = sum;
        }
    }

    #[kernel]
    pub fn ember_column_second_moment_kernel(
        column_partials: &[f32],
        mut column_second_moment: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        beta2: f32,
    ) {
        static mut REDUCE: SharedArray<f32, EMBER_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let col = thread::blockIdx_x();
        if col >= cols {
            return;
        }
        let row_tiles = rows.div_ceil(EMBER_COLUMN_ROW_TILE);
        let (thread, lane, warp) = thread_lane_warp();
        let mut sum = 0.0;
        let mut row_tile = thread;
        while row_tile < row_tiles {
            sum += column_partials[(row_tile * cols + col) as usize];
            row_tile += EMBER_THREADS_PER_BLOCK;
        }
        let sum = unsafe { block_sum_shared_f32(&mut REDUCE, sum, lane, warp) };
        if thread == 0 {
            let index = col as usize;
            unsafe {
                let old = *column_second_moment.as_mut_ptr().add(index);
                *column_second_moment.as_mut_ptr().add(index) =
                    beta2 * old + (1.0 - beta2) * (sum / rows as f32);
            }
        }
    }

    #[kernel]
    pub fn ember_normalizer_kernel(
        row_second_moment: &[f32],
        column_second_moment: &[f32],
        mut normalizer: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        beta2_correction: f32,
    ) {
        static mut REDUCE: SharedArray<f32, EMBER_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let (thread, lane, warp) = thread_lane_warp();
        let mut row_sum = 0.0;
        let mut row = thread;
        while row < rows {
            row_sum += row_second_moment[row as usize];
            row += EMBER_THREADS_PER_BLOCK;
        }
        let row_sum = unsafe { block_sum_shared_f32(&mut REDUCE, row_sum, lane, warp) };

        let mut column_sum = 0.0;
        let mut col = thread;
        while col < cols {
            column_sum += column_second_moment[col as usize];
            col += EMBER_THREADS_PER_BLOCK;
        }
        let column_sum = unsafe { block_sum_shared_f32(&mut REDUCE, column_sum, lane, warp) };

        if thread == 0 {
            let row_mean = row_sum / (rows as f32 * beta2_correction);
            let column_mean = column_sum / (cols as f32 * beta2_correction);
            unsafe {
                *normalizer.as_mut_ptr() = sqrt_f32(max_f32(row_mean * column_mean, 0.0));
            }
        }
    }

    #[kernel]
    pub fn ember_update_kernel(
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        grad: &[f32],
        row_second_moment: &[f32],
        column_second_moment: &[f32],
        normalizer: &[f32],
        rows: u32,
        cols: u32,
        grad_scale: f32,
        learning_rate: f32,
        weight_decay: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
    ) {
        let index = thread::blockIdx_x() * EMBER_THREADS_PER_BLOCK + thread::threadIdx_x();
        let len = rows * cols;
        if index >= len {
            return;
        }

        let row = index / cols;
        let col = index - row * cols;
        let row_second = row_second_moment[row as usize] / beta2_correction;
        let column_second = column_second_moment[col as usize] / beta2_correction;
        let scale = max_f32(normalizer[0], eps * eps);
        let factored_second = max_f32(row_second * column_second / scale, 0.0);
        let update = (grad[index as usize] * grad_scale) / (sqrt_f32(factored_second) + eps);

        unsafe {
            let z = z_master.as_mut_ptr().add(index as usize);
            let x = x_master.as_mut_ptr().add(index as usize);
            let next = *z * (1.0 - learning_rate * weight_decay) - learning_rate * update;
            *z = next;
            *x += average_coefficient * (next - *x);
        }
    }
}
