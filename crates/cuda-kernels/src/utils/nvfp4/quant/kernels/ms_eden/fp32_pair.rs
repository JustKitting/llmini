use cuda_device::{DisjointSlice, cuda_module, kernel};

#[path = "fp32_pair/dispatch.rs"]
mod dispatch;

use super::AMAX_WARPS_PER_BLOCK;
use super::body::{
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax, fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad,
    fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2,
};
use super::pack::ms_eden_pack_chunk_no_chunk_amax_row;
use super::random::random_sign;
use dispatch::{dispatch_fp32_pair, dispatch_fp32_pair_tiled};

const FP32_PAIR_TRANSPOSE_TILE_ROWS: usize = 32;
const FP32_PAIR_TRANSPOSE_TILE_COLS: usize = 8;
const FP32_PAIR_TRANSPOSE_TILE_STRIDE: usize = FP32_PAIR_TRANSPOSE_TILE_COLS + 1;
const FP32_PAIR_TRANSPOSE_TILE_ELEMS: usize =
    FP32_PAIR_TRANSPOSE_TILE_ROWS * FP32_PAIR_TRANSPOSE_TILE_STRIDE;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        transpose_dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        dispatch_fp32_pair!(
            row_grid_dim: row_grid_dim,
            x: x,
            output: [out_fp4, out_scales, out_global_scales],
            transpose_output: [transpose_out_fp4, transpose_out_scales, transpose_out_global_scales],
            scale: [global_scale, scale_override, sign_seed, scale_seed, transpose_scale_seed],
            row: fp32_to_nvfp4_ms_eden_body_no_chunk_amax(source_cols, dst_row_len);
            transpose: fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax(
                source_rows,
                transpose_dst_row_len,
                source_cols,
            )
        );
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row_shift: u32,
        transpose_chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        dispatch_fp32_pair!(
            row_grid_dim: row_grid_dim,
            x: x,
            output: [out_fp4, out_scales, out_global_scales],
            transpose_output: [transpose_out_fp4, transpose_out_scales, transpose_out_global_scales],
            scale: [global_scale, scale_override, sign_seed, scale_seed, transpose_scale_seed],
            row: fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
                source_cols,
                row_chunks_per_row_shift,
            );
            transpose: fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
                source_cols,
                transpose_chunks_per_row_shift,
            )
        );
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_pow2_tiled_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row_shift: u32,
        transpose_chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        dispatch_fp32_pair_tiled!(
            row_grid_dim: row_grid_dim,
            source_rows: 32u32 << transpose_chunks_per_row_shift,
            source_cols: source_cols,
            transpose_chunks: transpose_chunks_per_row_shift,
            transpose_chunks_are_shift: true,
            x: x,
            output: [out_fp4, out_scales, out_global_scales],
            transpose_output: [transpose_out_fp4, transpose_out_scales, transpose_out_global_scales],
            scale: [global_scale, scale_override, sign_seed, scale_seed, transpose_scale_seed],
            row: fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad_pow2(
                source_cols,
                row_chunks_per_row_shift,
            )
        );
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row: u32,
        transpose_chunks_per_row: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        dispatch_fp32_pair!(
            row_grid_dim: row_grid_dim,
            x: x,
            output: [out_fp4, out_scales, out_global_scales],
            transpose_output: [transpose_out_fp4, transpose_out_scales, transpose_out_global_scales],
            scale: [global_scale, scale_override, sign_seed, scale_seed, transpose_scale_seed],
            row: fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(source_cols, row_chunks_per_row);
            transpose: fp32_transpose_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
                source_cols,
                transpose_chunks_per_row,
            )
        );
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_tiled_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        row_grid_dim: u32,
        source_cols: u32,
        row_chunks_per_row: u32,
        transpose_chunks_per_row: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
        transpose_scale_seed: u32,
    ) {
        dispatch_fp32_pair_tiled!(
            row_grid_dim: row_grid_dim,
            source_rows: transpose_chunks_per_row * 32,
            source_cols: source_cols,
            transpose_chunks: transpose_chunks_per_row,
            transpose_chunks_are_shift: false,
            x: x,
            output: [out_fp4, out_scales, out_global_scales],
            transpose_output: [transpose_out_fp4, transpose_out_scales, transpose_out_global_scales],
            scale: [global_scale, scale_override, sign_seed, scale_seed, transpose_scale_seed],
            row: fp32_to_nvfp4_ms_eden_body_no_chunk_amax_no_pad(
                source_cols,
                row_chunks_per_row,
            )
        );
    }
}
