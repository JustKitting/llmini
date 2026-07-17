use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use super::HADAMARD_DIM;
use super::input::nvfp4_transposed_hadamard_input;
use super::pack::{
    guarded_pack_chunk, ms_eden_pack_chunk, ms_eden_pack_chunk_no_chunk_amax,
    ms_eden_pack_chunk_no_chunk_amax_row, pack_chunk,
};
use super::random::random_sign;
use super::transpose_kernels::pack_padded_transpose_chunk;
use crate::nvfp4::nvfp4_values2;

const TRANSPOSE_TILE_ROWS: usize = 32;
const TRANSPOSE_TILE_COLS: usize = 8;
const TRANSPOSE_TILE_STRIDE: usize = TRANSPOSE_TILE_COLS + 1;
const TRANSPOSE_TILE_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_STRIDE;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        guarded_pack_chunk!(chunk, chunk_count);
        pack_padded_transpose_chunk!(
            chunk_amax,
            input: nvfp4_transposed_hadamard_input(bytes, scales, source_global_scale),
            chunk: chunk,
            output: [
                &mut out_fp4,
                &mut out_scales,
                &mut out_global_scales,
                &mut out_chunk_amax
            ],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        guarded_pack_chunk!(chunk, chunk_count);
        pack_padded_transpose_chunk!(
            no_chunk_amax,
            input: nvfp4_transposed_hadamard_input(bytes, scales, source_global_scale),
            chunk: chunk,
            output: [&mut out_fp4, &mut out_scales, &mut out_global_scales],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        pack_padded_transpose_chunk!(
            no_chunk_amax,
            input: nvfp4_transposed_hadamard_input(bytes, scales, source_global_scale),
            chunk: pack_chunk(),
            output: [&mut out_fp4, &mut out_scales, &mut out_global_scales],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }

    #[kernel]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
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

        if tile_col & 1 == 0 {
            let index = ((source_row << source_cols_shift) + source_col) as usize;
            let (lo, hi) = nvfp4_values2(bytes, scales, source_global_scale[0], index);
            unsafe {
                TILE[tile_row * TRANSPOSE_TILE_STRIDE + tile_col] = lo;
                TILE[tile_row * TRANSPOSE_TILE_STRIDE + tile_col + 1] = hi;
            }
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

    #[kernel]
    pub fn nvfp4_transpose_to_nvfp4_ms_eden_device_scale_no_chunk_amax_exact_no_pad_source_cols_pow2_tiled_row_mul_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scale: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scales: DisjointSlice<f32>,
        global_scale: &[f32],
        source_cols_shift: u32,
        chunks_per_row: u32,
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

        if tile_col & 1 == 0 {
            let index = ((source_row << source_cols_shift) + source_col) as usize;
            let (lo, hi) = nvfp4_values2(bytes, scales, source_global_scale[0], index);
            unsafe {
                TILE[tile_row * TRANSPOSE_TILE_STRIDE + tile_col] = lo;
                TILE[tile_row * TRANSPOSE_TILE_STRIDE + tile_col + 1] = hi;
            }
        }
        thread::sync_threads();

        let lane = warp::lane_id();
        let warp_in_block = thread::threadIdx_x() / 32;
        let row = source_col_base + warp_in_block;
        let source_row_chunk = thread::blockIdx_x();
        let chunk = row * chunks_per_row + source_row_chunk;
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
