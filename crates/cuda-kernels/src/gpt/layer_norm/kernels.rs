use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};

use super::{
    ROW_SIZE,
    body::{gpt_layer_norm_body, maybe_store_residual_f16},
};
use crate::float_ptx::{fma_f32, sqrt_f32};
use crate::warp_reduce::warp_sum_f32;

const ROW_SIZE_F32: f32 = ROW_SIZE as f32;

pub use module::{LoadedModule, from_module};

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn layer_norm_warp_f32_kernel(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        epsilon: f32,
    ) {
        let lane = warp::lane_id() as usize;
        let warp_in_block = thread::threadIdx_x() / ROW_SIZE as u32;
        let warps_per_block = thread::blockDim_x() / ROW_SIZE as u32;
        let row = thread::blockIdx_x() * warps_per_block + warp_in_block;

        if row < row_count {
            let index = row as usize * ROW_SIZE + lane;
            let value = x[index];
            let mean = warp_sum_f32(value) / ROW_SIZE_F32;
            let centered = value - mean;
            let variance = warp_sum_f32(centered * centered) / ROW_SIZE_F32;
            let inv_std = 1.0 / sqrt_f32(variance + epsilon);
            let normalized = centered * inv_std;

            unsafe {
                *out.get_unchecked_mut(index) = fma_f32(normalized, gamma[lane], beta[lane]);
            }
        }
    }

    #[kernel]
    pub fn gpt_layer_norm_kernel(
        residual: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        row_count: u32,
        embedding_dim: u32,
        epsilon: f32,
        output_scale: f32,
    ) {
        gpt_layer_norm_body!(
            residual weight_bytes weight_scales bias_bytes bias_scales;
            weight_global_scale bias_global_scale;
            normalized normalized_amax mean_out inv_std_out;
            row_count embedding_dim epsilon output_scale;
            none
        );
    }

    #[kernel]
    pub fn gpt_layer_norm_save_residual_f16_kernel(
        residual: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut normalized: DisjointSlice<f32>,
        mut normalized_amax: DisjointSlice<f32>,
        mut mean_out: DisjointSlice<f32>,
        mut inv_std_out: DisjointSlice<f32>,
        mut residual_f16: DisjointSlice<u16>,
        row_count: u32,
        embedding_dim: u32,
        epsilon: f32,
        output_scale: f32,
    ) {
        gpt_layer_norm_body!(
            residual weight_bytes weight_scales bias_bytes bias_scales;
            weight_global_scale bias_global_scale;
            normalized normalized_amax mean_out inv_std_out;
            row_count embedding_dim epsilon output_scale;
            residual_f16
        );
    }
}
