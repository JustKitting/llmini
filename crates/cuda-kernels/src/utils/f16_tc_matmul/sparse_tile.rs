use super::convert::load_f32_global_read_only;
use super::cta_tile::{CTA_M, CTA_N};

#[inline(always)]
pub(crate) fn attention_tile_count(seq_len: u32) -> u32 {
    seq_len.div_ceil(CTA_M)
}

#[inline(always)]
pub(crate) fn attention_tile_scale(
    scales: &[f32],
    batch_head: u32,
    seq_len: u32,
    query_base: u32,
    key_base: u32,
) -> f32 {
    let tiles = attention_tile_count(seq_len);
    let query_tile = query_base / CTA_M;
    let key_tile = key_base / CTA_N;
    let index = ((batch_head * tiles + query_tile) * tiles + key_tile) as usize;
    load_f32_global_read_only(scales.as_ptr(), index)
}
