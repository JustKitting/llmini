use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::nvfp4::nvfp4_rowwise_value;

use super::super::input::{no_pad_pow2_chunk_position, nvfp4_rowwise_value_at_pow2};
use super::super::pack::{ms_eden_pack_chunk_no_chunk_amax_row, pack_chunk};
use super::super::random::random_sign;

const TRANSPOSE_TILE_ROWS: usize = 32;
const TRANSPOSE_TILE_COLS: usize = 8;
const TRANSPOSE_TILE_STRIDE: usize = TRANSPOSE_TILE_COLS + 1;
const TRANSPOSE_TILE_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_STRIDE;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let (row, input_col, first_chunk_in_row) =
            no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
        let input = nvfp4_rowwise_value(
            bytes,
            scales,
            source_global_scales,
            source_cols as usize,
            input_col as usize,
            row as usize,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            first_chunk_in_row,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols_shift: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        let lane = warp::lane_id();
        let chunk = pack_chunk();
        let (row, input_col, first_chunk_in_row) =
            no_pad_pow2_chunk_position(chunk, lane, chunks_per_row_shift);
        let input = nvfp4_rowwise_value_at_pow2(
            bytes,
            scales,
            source_global_scales,
            source_cols_shift,
            input_col,
            row,
        ) * random_sign(sign_seed, input_col);
        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            first_chunk_in_row,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }

    #[kernel]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols_shift: u32,
        chunks_per_row_shift: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        static mut TILE: SharedArray<f32, TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let tile_row = thread_id / TRANSPOSE_TILE_COLS;
        let tile_col = thread_id - tile_row * TRANSPOSE_TILE_COLS;
        let source_row_base = thread::blockIdx_x() * TRANSPOSE_TILE_ROWS as u32;
        let source_col_base = thread::blockIdx_y() * TRANSPOSE_TILE_COLS as u32;
        let source_row = source_row_base + tile_row as u32;
        let source_col = source_col_base + tile_col as u32;

        unsafe {
            TILE[tile_row * TRANSPOSE_TILE_STRIDE + tile_col] = nvfp4_rowwise_value_at_pow2(
                bytes,
                scales,
                source_global_scales,
                source_cols_shift,
                source_row,
                source_col,
            );
        }
        thread::sync_threads();

        let lane = warp::lane_id();
        let warp_in_block = thread::threadIdx_x() / 32;
        let row = source_col_base + warp_in_block;
        let source_row_chunk = thread::blockIdx_x();
        let chunk = (row << chunks_per_row_shift) + source_row_chunk;
        let input_col = source_row_base + lane;
        let input = unsafe { TILE[lane as usize * TRANSPOSE_TILE_STRIDE + warp_in_block as usize] }
            * random_sign(sign_seed, input_col);

        ms_eden_pack_chunk_no_chunk_amax_row(
            input,
            &mut out_fp4,
            &mut out_scales,
            &mut out_global_scales,
            chunk,
            row,
            source_row_chunk == 0,
            global_scale[0],
            scale_override,
            scale_seed,
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
