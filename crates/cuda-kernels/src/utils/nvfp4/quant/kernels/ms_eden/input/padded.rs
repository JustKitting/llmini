use crate::f16_tc_matmul::convert::load_f32_global_read_only;
use crate::nvfp4::nvfp4_rowwise_value;

use super::input_position::padded_chunk_position;
use super::input_values::nvfp4_value_at;
use super::random::random_sign;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn nvfp4_transposed_hadamard_input(
    bytes: &[u8],
    scales: &[u8],
    global_scale: &[f32],
    chunk_base: u32,
    lane: u32,
    source_rows: u32,
    source_cols: u32,
    dst_row_len: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < source_rows {
        let source_index = input_col * source_cols + row;
        nvfp4_value_at(bytes, scales, global_scale, source_index)
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub(super) fn rowwise_transposed_hadamard_input(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    chunk_base: u32,
    lane: u32,
    source_rows: u32,
    source_cols: u32,
    dst_row_len: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < source_rows {
        nvfp4_rowwise_value(
            bytes,
            scales,
            global_scales,
            source_cols as usize,
            input_col as usize,
            row as usize,
        )
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[inline(always)]
pub(super) fn hadamard_input(
    x: &[f32],
    chunk_base: u32,
    lane: u32,
    src_row_len: u32,
    dst_row_len: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < src_row_len {
        let index = row * src_row_len + input_col;
        load_f32_global_read_only(x.as_ptr(), index as usize)
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}

#[inline(always)]
pub(super) fn transposed_hadamard_input(
    x: &[f32],
    chunk_base: u32,
    lane: u32,
    source_rows: u32,
    dst_row_len: u32,
    source_cols: u32,
    seed: u32,
) -> f32 {
    let (row, input_col) = padded_chunk_position(chunk_base, lane, dst_row_len);
    let input = if input_col < source_rows {
        let index = input_col * source_cols + row;
        load_f32_global_read_only(x.as_ptr(), index as usize)
    } else {
        0.0
    };
    input * random_sign(seed, input_col)
}
