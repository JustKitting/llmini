#[inline(always)]
pub(super) fn padded_chunk_position(chunk_base: u32, lane: u32, dst_row_len: u32) -> (u32, u32) {
    let row = chunk_base / dst_row_len;
    let row_base = row * dst_row_len;
    (row, chunk_base - row_base + lane)
}
