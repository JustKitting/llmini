use crate::attention::CausalAttentionParams;
pub(crate) use crate::attention::layout::{
    compact_index, compact_linear_parts, hidden_index, qkv_index,
};

use super::activation::beta_offset;
use super::shape::chunk_count;

/// Each chunk owns enough bytes for FP32 U. Its native FP16 Kpos occupies
/// the first half of those bytes until the joint W/U kernel overwrites it.
/// Keeping that ownership stride prevents expansion into another CTA's input.
#[inline(always)]
pub(crate) fn kpos_half_arena_index(compact: usize) -> usize {
    const CHUNK_ELEMS: usize = 64 * 64;
    (compact / CHUNK_ELEMS) * (2 * CHUNK_ELEMS) + compact % CHUNK_ELEMS
}

#[inline(always)]
pub(crate) fn beta_compact_index(
    batch: u32,
    token: u32,
    head: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((batch * params.head_count + head) * params.seq_len + token) as usize
}

#[inline(always)]
pub(crate) fn chunk_g_last_index(
    bh: u32,
    chunk: u32,
    dim: u32,
    params: &CausalAttentionParams,
) -> usize {
    ((bh * chunk_count(params) + chunk) * params.head_dim + dim) as usize
}

#[inline(always)]
pub(crate) fn beta_index(row: u32, head: u32, params: &CausalAttentionParams) -> usize {
    row as usize * params.qkv_dim as usize + beta_offset(params) as usize + head as usize
}
