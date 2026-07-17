use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::float_ptx::{abs_f32, max_f32};
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{
    f16_column, f32_column, layer_norm_columns3, layer_norm_map3, layer_norm_map3_indexed,
    layer_norm_store3, layer_norm_sum3, max_abs3, nvfp4_column,
};
use crate::warp_reduce::{thread_lane_warp, warp_max_f32, warp_sum_f32};

pub const THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const WARPS_PER_BLOCK: u32 = THREADS_PER_BLOCK / WARP_SIZE;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod kernels {
    use super::*;

    macro_rules! maybe_store_input_amax {
        (none; $($ignored:ident),+) => {};
        ($chunk_amax:ident; $dx:ident, $cols:ident, $d_residual:ident, $row_base:ident, $row:ident, $embedding_dim:ident, $thread:ident, $lane:ident, $warp_in_block:ident) => {{
            let valid_dx = layer_norm_map3_indexed!($dx, |index, value| {
                if $cols[index] < $embedding_dim {
                    value
                } else {
                    0.0
                }
            });
            let mut local_amax = max_abs3(valid_dx[0], valid_dx[1], valid_dx[2]);
            let mut col = $thread + THREADS_PER_BLOCK * 3;
            while col < $embedding_dim {
                let value = unsafe { *$d_residual.as_mut_ptr().add($row_base + col as usize) };
                local_amax = max_f32(local_amax, abs_f32(value));
                col += THREADS_PER_BLOCK;
            }
            let row_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                local_amax,
                $lane,
                $warp_in_block,
                warp_max_f32
            );
            layer_norm_store_row!(&mut $chunk_amax, $row, $lane, $warp_in_block, row_amax);
        }};
    }

    macro_rules! layer_norm_backward_input_body {
        (
            $residual_column:path;
            $residual:ident $d_normalized:ident $mean:ident $inv_std:ident;
            $weight_bytes:ident $weight_scales:ident $weight_global_scale:ident;
            $d_residual:ident $output_scale:ident $row_count:ident $embedding_dim:ident;
            $direct:expr;
            $chunk_amax:ident
        ) => {{
            static mut WARP_SUMS: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
                SharedArray::UNINIT;

            let row = thread::blockIdx_x();
            let (thread, lane, warp_in_block) = thread_lane_warp();

            if row < $row_count {
                let row_base = row as usize * $embedding_dim as usize;
                let cols = layer_norm_columns3!(thread, THREADS_PER_BLOCK);
                let row_mean = $mean[row as usize];
                let row_inv_std = $inv_std[row as usize];
                let xhat = layer_norm_map3!(cols, |col| {
                    ($residual_column($residual, row_base, col, $embedding_dim) - row_mean)
                        * row_inv_std
                });
                let dxhat = layer_norm_map3!(cols, |col| {
                    let grad = f32_column($d_normalized, row_base, col, $embedding_dim);
                    let weight = nvfp4_column(
                        $weight_bytes,
                        $weight_scales,
                        $weight_global_scale[0],
                        0,
                        col,
                        $embedding_dim,
                    );
                    grad * weight * $output_scale
                });
                let dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    layer_norm_sum3!(dxhat),
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let xhat_dxhat =
                    layer_norm_map3_indexed!(xhat, |index, value| value * dxhat[index]);
                let xhat_dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    layer_norm_sum3!(xhat_dxhat),
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let inv_dim = 1.0 / $embedding_dim as f32;
                let dx = layer_norm_map3_indexed!(dxhat, |index, value| {
                    (value - dxhat_sum * inv_dim - xhat[index] * xhat_dxhat_sum * inv_dim)
                        * row_inv_std
                });
                let direct: *const f32 = $direct;
                let dx = if direct.is_null() {
                    dx
                } else {
                    layer_norm_map3_indexed!(dx, |index, value| {
                        let col = cols[index];
                        if col < $embedding_dim {
                            unsafe { *direct.add(row_base + col as usize) + value }
                        } else {
                            value
                        }
                    })
                };

                layer_norm_store3!(&mut $d_residual, row_base, cols, $embedding_dim, dx);
                maybe_store_input_amax!(
                    $chunk_amax;
                    dx,
                    cols,
                    $d_residual,
                    row_base,
                    row,
                    $embedding_dim,
                    thread,
                    lane,
                    warp_in_block
                );
            }
        }};
    }

    #[kernel]
    pub fn layer_norm_backward_input_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            none
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_amax_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            chunk_amax
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_add_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            none
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_add_amax_kernel(
        residual: &[u16],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        direct: &[f32],
        mut d_residual: DisjointSlice<f32>,
        mut chunk_amax: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f16_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            direct.as_ptr();
            chunk_amax
        );
    }

    #[kernel]
    pub fn layer_norm_backward_input_f32_kernel(
        residual: &[f32],
        d_normalized: &[f32],
        mean: &[f32],
        inv_std: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        weight_global_scale: &[f32],
        mut d_residual: DisjointSlice<f32>,
        output_scale: f32,
        row_count: u32,
        embedding_dim: u32,
    ) {
        layer_norm_backward_input_body!(
            f32_column;
            residual d_normalized mean inv_std;
            weight_bytes weight_scales weight_global_scale;
            d_residual output_scale row_count embedding_dim;
            core::ptr::null();
            none
        );
    }
}
