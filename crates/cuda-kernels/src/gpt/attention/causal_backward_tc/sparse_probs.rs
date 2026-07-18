use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::attention::CausalAttentionParams;
use crate::block_reduce::block_sum_shared_f32;
use crate::f16_tc_matmul::convert::{cvt_f32_f16, load_f16_global_bits_read_only};
use crate::f16_tc_matmul::cta_tile::{CTA_M, CTA_N};
use crate::warp_reduce::thread_lane_warp;

pub(super) const SPARSE_PROB_THREADS_PER_BLOCK: u32 = 256;
const SPARSE_PROB_WARPS_PER_BLOCK: usize = (SPARSE_PROB_THREADS_PER_BLOCK / 32) as usize;
const TILE_ELEMENTS: u32 = CTA_M * CTA_N;
const U24_TO_UNIT_F32: f32 = 1.0 / 16_777_216.0;

pub(super) fn sparsify_attention_probs_f16_body(
    probs: &[u16],
    mut tile_scales: DisjointSlice<f32>,
    seed: u32,
    tile_budget: f32,
    params: CausalAttentionParams,
    reduce: &mut SharedArray<f32, SPARSE_PROB_WARPS_PER_BLOCK>,
) {
    let key_tile = thread::blockIdx_x();
    let query_tile = thread::blockIdx_y();
    let batch_head = thread::blockIdx_z();
    let tiles = params.seq_len.div_ceil(CTA_M);
    let window_tiles = params.attention_window / CTA_N;
    if batch_head >= params.batch_size * params.head_count
        || query_tile >= tiles
        || key_tile >= tiles
        || key_tile > query_tile
        || query_tile > key_tile + window_tiles
    {
        return;
    }

    let thread_id = thread::threadIdx_x();
    let mut local_index = thread_id;
    let mut local_mass = 0.0;
    let query_base = query_tile * CTA_M;
    let key_base = key_tile * CTA_N;
    while local_index < TILE_ELEMENTS {
        let local_row = local_index / CTA_N;
        let local_col = local_index - local_row * CTA_N;
        let row = query_base + local_row;
        let col = key_base + local_col;
        if row < params.seq_len && col < params.seq_len {
            let index = ((batch_head * params.seq_len + row) * params.seq_len + col) as usize;
            local_mass += cvt_f32_f16(load_f16_global_bits_read_only(probs.as_ptr(), index));
        }
        local_index += SPARSE_PROB_THREADS_PER_BLOCK;
    }

    let (_, lane, warp_in_block) = thread_lane_warp();
    let tile_mass = block_sum_shared_f32(reduce, local_mass, lane, warp_in_block);
    if thread_id == 0 {
        let q_unclamped = tile_budget * tile_mass / CTA_M as f32;
        let q = if q_unclamped < 1.0 { q_unclamped } else { 1.0 };
        let tile_index = (batch_head * tiles + query_tile) * tiles + key_tile;
        let random =
            (hash_u32(seed ^ tile_index.wrapping_mul(0x9e37_79b9)) >> 8) as f32 * U24_TO_UNIT_F32;
        let scale = if q > 0.0 && random < q { 1.0 / q } else { 0.0 };
        unsafe {
            *tile_scales.get_unchecked_mut(tile_index as usize) = scale;
        }
    }
}

#[inline(always)]
fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}
