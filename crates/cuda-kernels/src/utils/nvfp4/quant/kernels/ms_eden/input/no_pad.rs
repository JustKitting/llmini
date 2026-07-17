use super::input_position::{no_pad_chunk_position, no_pad_pow2_chunk_position};
use super::random::random_sign;
use crate::f16_tc_matmul::convert::load_f32_global_read_only;

macro_rules! no_pad_hadamard_input_fn {
    ($name:ident, $position_fn:ident, $row_len_arg:ident, $chunks_arg:ident, |$row:ident, $col:ident| $index:expr) => {
        #[inline(always)]
        pub(super) fn $name(
            x: &[f32],
            chunk: u32,
            lane: u32,
            $row_len_arg: u32,
            $chunks_arg: u32,
            seed: u32,
        ) -> (f32, u32, bool) {
            let ($row, $col, first_chunk_in_row) = $position_fn(chunk, lane, $chunks_arg);
            no_pad_result(x, $index, seed, ($row, $col, first_chunk_in_row))
        }
    };
}

no_pad_hadamard_input_fn!(
    hadamard_input_no_pad_pow2,
    no_pad_pow2_chunk_position,
    src_row_len,
    chunks_per_row_shift,
    |row, col| row * src_row_len + col
);
no_pad_hadamard_input_fn!(
    transposed_hadamard_input_no_pad_pow2,
    no_pad_pow2_chunk_position,
    source_cols,
    chunks_per_row_shift,
    |row, col| col * source_cols + row
);
no_pad_hadamard_input_fn!(
    hadamard_input_no_pad,
    no_pad_chunk_position,
    src_row_len,
    chunks_per_row,
    |row, col| row * src_row_len + col
);
no_pad_hadamard_input_fn!(
    transposed_hadamard_input_no_pad,
    no_pad_chunk_position,
    source_cols,
    chunks_per_row,
    |row, col| col * source_cols + row
);

#[inline(always)]
fn no_pad_result(x: &[f32], index: u32, seed: u32, position: (u32, u32, bool)) -> (f32, u32, bool) {
    let (row, input_col, first_chunk_in_row) = position;
    (
        load_f32_global_read_only(x.as_ptr(), index as usize) * random_sign(seed, input_col),
        row,
        first_chunk_in_row,
    )
}
