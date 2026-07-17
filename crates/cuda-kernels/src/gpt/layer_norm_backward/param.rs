use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::atomic::atomic_add_f32;
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{f16_column, f32_column};
use crate::warp_reduce::{thread_lane_warp, warp_sum_f32};

pub const PARAM_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = PARAM_THREADS_PER_BLOCK / WARP_SIZE;
const ROWS_PER_THREAD: u32 = 4;
const UNROLLED_ROW_STRIDE: u32 = PARAM_THREADS_PER_BLOCK * ROWS_PER_THREAD;
pub const PARAM_TILE_COLS: u32 = WARP_SIZE;
pub const PARAM_ROW_PARTITIONS: u32 = 16;
const PARAM_TILED_ROW_STRIDE: u32 = WARPS_PER_BLOCK * PARAM_ROW_PARTITIONS;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod kernels {
    use super::*;

    macro_rules! layer_norm_backward_params_body {
        (
            $residual_column:path;
            $residual:ident $d_normalized:ident $mean:ident $inv_std:ident;
            $d_weight:ident $d_bias:ident $output_scale:ident $row_count:ident $embedding_dim:ident
        ) => {{
            static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
                SharedArray::UNINIT;

            let col = thread::blockIdx_x();
            let (tid, lane, warp_in_block) = thread_lane_warp();

            if col < $embedding_dim {
                let mut weight_local = 0.0f32;
                let mut bias_local = 0.0f32;
                let mut row = tid;

                macro_rules! accumulate_param_grad {
                    ($weight:ident, $bias:ident, $row:expr) => {{
                        let row = $row;
                        let offset = row as usize * $embedding_dim as usize + col as usize;
                        let grad = $d_normalized[offset] * $output_scale;
                        let row_base = row as usize * $embedding_dim as usize;
                        let xhat = ($residual_column($residual, row_base, col, $embedding_dim)
                            - $mean[row as usize])
                            * $inv_std[row as usize];
                        $weight += grad * xhat;
                        $bias += grad;
                    }};
                }

                while row + PARAM_THREADS_PER_BLOCK * 3 < $row_count {
                    accumulate_param_grad!(weight_local, bias_local, row);
                    accumulate_param_grad!(weight_local, bias_local, row + PARAM_THREADS_PER_BLOCK);
                    accumulate_param_grad!(
                        weight_local,
                        bias_local,
                        row + PARAM_THREADS_PER_BLOCK * 2
                    );
                    accumulate_param_grad!(
                        weight_local,
                        bias_local,
                        row + PARAM_THREADS_PER_BLOCK * 3
                    );
                    row += UNROLLED_ROW_STRIDE;
                }

                while row < $row_count {
                    accumulate_param_grad!(weight_local, bias_local, row);
                    row += PARAM_THREADS_PER_BLOCK;
                }

                let weight_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    weight_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let bias_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    bias_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );

                layer_norm_store_row!(&mut $d_weight, col, lane, warp_in_block, weight_sum);
                layer_norm_store_row!(&mut $d_bias, col, lane, warp_in_block, bias_sum);
            }
        }};
    }

    #[kernel]
    pub fn layer_norm_backward_params_tiled_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        mut d_weight: DisjointSlice<f32>,
        mut d_bias: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        static mut WEIGHT_PARTIALS: SharedArray<f32, { PARAM_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        static mut BIAS_PARTIALS: SharedArray<f32, { PARAM_THREADS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let partition = block % PARAM_ROW_PARTITIONS;
        let col_tile = block / PARAM_ROW_PARTITIONS;
        let (tid, lane, warp_in_block) = thread_lane_warp();
        let col = col_tile * PARAM_TILE_COLS + lane;
        let mut weight_local = 0.0f32;
        let mut bias_local = 0.0f32;
        let mut row = partition * WARPS_PER_BLOCK + warp_in_block;

        if col < embedding_dim {
            while row < row_count {
                let row_base = row as usize * embedding_dim as usize;
                let grad = d_normalized[row_base + col as usize] * output_scale;
                let row_mean =
                    warp::shuffle_f32(if lane == 0 { mean[row as usize] } else { 0.0 }, 0);
                let row_inv_std = warp::shuffle_f32(
                    if lane == 0 {
                        inv_std[row as usize]
                    } else {
                        0.0
                    },
                    0,
                );
                let xhat =
                    (f16_column(residual, row_base, col, embedding_dim) - row_mean) * row_inv_std;
                weight_local += grad * xhat;
                bias_local += grad;
                row += PARAM_TILED_ROW_STRIDE;
            }
        }

        unsafe {
            WEIGHT_PARTIALS[tid as usize] = weight_local;
            BIAS_PARTIALS[tid as usize] = bias_local;
        }
        thread::sync_threads();

        if warp_in_block == 0 && col < embedding_dim {
            let mut weight_sum = 0.0f32;
            let mut bias_sum = 0.0f32;
            let mut partial_warp = 0u32;
            while partial_warp < WARPS_PER_BLOCK {
                let index = (partial_warp * WARP_SIZE + lane) as usize;
                unsafe {
                    weight_sum += WEIGHT_PARTIALS[index];
                    bias_sum += BIAS_PARTIALS[index];
                }
                partial_warp += 1;
            }
            unsafe {
                atomic_add_f32(d_weight.as_mut_ptr().add(col as usize), weight_sum);
                atomic_add_f32(d_bias.as_mut_ptr().add(col as usize), bias_sum);
            }
        }
    }

    #[kernel]
    pub fn layer_norm_backward_params_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        mut d_weight: DisjointSlice<f32>,
        mut d_bias: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_params_body!(
            f16_column;
            residual d_normalized mean inv_std;
            d_weight d_bias output_scale row_count embedding_dim
        );
    }

    #[kernel]
    pub fn layer_norm_backward_params_f32_kernel(
        residual: &[f32],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        mut d_weight: DisjointSlice<f32>,
        mut d_bias: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_params_body!(
            f32_column;
            residual d_normalized mean inv_std;
            d_weight d_bias output_scale row_count embedding_dim
        );
    }
}
