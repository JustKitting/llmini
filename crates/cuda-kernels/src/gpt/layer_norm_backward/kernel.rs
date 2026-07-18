use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::float_ptx::{abs_f32, max_f32};
use crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
use crate::layer_norm_utils::{f16_column, f32_column, nvfp4_column};
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
        ($chunk_amax:ident; $local_amax:ident, $row:ident, $lane:ident, $warp_in_block:ident) => {{
            let row_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                WARPS_PER_BLOCK,
                $local_amax,
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
                let row_mean = $mean[row as usize];
                let row_inv_std = $inv_std[row as usize];
                let mut dxhat_local = 0.0f32;
                let mut xhat_dxhat_local = 0.0f32;
                let mut col = thread;
                while col < $embedding_dim {
                    let xhat = ($residual_column(
                        $residual,
                        row_base,
                        col,
                        $embedding_dim,
                    ) - row_mean)
                        * row_inv_std;
                    let grad = f32_column($d_normalized, row_base, col, $embedding_dim);
                    let weight = nvfp4_column(
                        $weight_bytes,
                        $weight_scales,
                        $weight_global_scale[0],
                        0,
                        col,
                        $embedding_dim,
                    );
                    let dxhat = grad * weight * $output_scale;
                    dxhat_local += dxhat;
                    xhat_dxhat_local += xhat * dxhat;
                    col += THREADS_PER_BLOCK;
                }
                let dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    dxhat_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let xhat_dxhat_sum = layer_norm_block_reduce!(
                    WARP_SUMS,
                    WARPS_PER_BLOCK,
                    xhat_dxhat_local,
                    lane,
                    warp_in_block,
                    warp_sum_f32
                );
                let inv_dim = 1.0 / $embedding_dim as f32;
                let direct: *const f32 = $direct;
                let mut local_amax = 0.0f32;
                let mut col = thread;
                while col < $embedding_dim {
                    let xhat = ($residual_column(
                        $residual,
                        row_base,
                        col,
                        $embedding_dim,
                    ) - row_mean)
                        * row_inv_std;
                    let grad = f32_column($d_normalized, row_base, col, $embedding_dim);
                    let weight = nvfp4_column(
                        $weight_bytes,
                        $weight_scales,
                        $weight_global_scale[0],
                        0,
                        col,
                        $embedding_dim,
                    );
                    let dxhat = grad * weight * $output_scale;
                    let mut dx = (dxhat
                        - dxhat_sum * inv_dim
                        - xhat * xhat_dxhat_sum * inv_dim)
                        * row_inv_std;
                    if !direct.is_null() {
                        dx += unsafe { *direct.add(row_base + col as usize) };
                    }
                    unsafe {
                        *$d_residual.get_unchecked_mut(row_base + col as usize) = dx;
                    }
                    local_amax = max_f32(local_amax, abs_f32(dx));
                    col += THREADS_PER_BLOCK;
                }
                maybe_store_input_amax!(
                    $chunk_amax;
                    local_amax,
                    row,
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
