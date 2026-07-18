use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};

use crate::amax::max4_f32;
use crate::block_reduce::{block_max_store_f32, block_sum_shared_f32};
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{abs_f32, max_f32};
use crate::mma::{
    Nvfp4ProjectionParams, dispatch_projection_cta_tiles, nvfp4_projection_cta_kernel_body,
    nvfp4_projection_cta_kernel_body_at_aligned_row_pair, nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
};
use crate::nvfp4::nvfp4_values2;
use crate::warp_reduce::thread_lane_warp;

pub(super) const RELU2_THREADS_PER_BLOCK: u32 = 256;
pub(super) const RELU2_VALUES_PER_BLOCK: u32 = 8 * RELU2_THREADS_PER_BLOCK;
const RELU2_WARPS_PER_BLOCK: usize = (RELU2_THREADS_PER_BLOCK / 32) as usize;
pub(super) const BLOCK_TOPK_TILE: u32 = 128;
pub(super) const BLOCK_TOPK_FEATURE_TILES: u32 = 64;
pub(super) const BLOCK_TOPK_TOKEN_TILES: u32 = 64;
pub(super) const BLOCK_TOPK_ACTIVE_FEATURE_TILES: u32 = 48;
pub(super) const BLOCK_TOPK_SCORE_THREADS: u32 = 256;
pub(super) const BLOCK_TOPK_MASK_THREADS: u32 = 1024;
const BLOCK_TOPK_SCORE_WARPS: usize = (BLOCK_TOPK_SCORE_THREADS / 32) as usize;
const BLOCK_TOPK_MASK_WARPS: u32 = BLOCK_TOPK_MASK_THREADS / 32;
const FULL_WARP_MASK: u32 = u32::MAX;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn mlp_projection_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        let params = params.with_global_scales(weight_global_scale[0], bias_global_scale[0]);

        dispatch_projection_cta_tiles!(
            params,
            nvfp4_projection_cta_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_kernel_body;
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, bias_bytes, bias_scales,
            &mut out, params,
        );
    }

    #[kernel]
    pub fn mlp_projection_relu2_kernel(
        input_bytes: &[u8],
        input_scales: &[u8],
        input_global_scales: &[f32],
        weight_bytes: &[u8],
        weight_scales: &[u8],
        bias_bytes: &[u8],
        bias_scales: &[u8],
        weight_global_scale: &[f32],
        bias_global_scale: &[f32],
        mut pre_activation: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        params: Nvfp4ProjectionParams,
    ) {
        let params = params.with_global_scales(weight_global_scale[0], bias_global_scale[0]);

        dispatch_projection_cta_tiles!(
            params,
            nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
            nvfp4_projection_cta_relu2_kernel_body;
            input_bytes, input_scales, input_global_scales,
            weight_bytes, weight_scales, bias_bytes, bias_scales,
            &mut pre_activation, &mut out, params,
        );
    }

    #[kernel]
    pub fn relu2_backward_kernel(
        pre_activation: &[f32],
        d_out: &[f32],
        mut d_pre_activation: DisjointSlice<f32>,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * super::RELU2_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let relu = max_f32(pre_activation[index as usize], 0.0);
            unsafe {
                *d_pre_activation.get_unchecked_mut(index as usize) =
                    d_out[index as usize] * 2.0 * relu;
            }
        }
    }

    #[kernel]
    pub fn relu2_backward_f16_kernel(
        pre_activation: &[u16],
        d_out: &[f32],
        mut d_pre_activation: DisjointSlice<f32>,
        mut d_pre_activation_chunk_amax: DisjointSlice<f32>,
        len: u32,
    ) {
        static mut AMAX: SharedArray<f32, RELU2_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let chunk = thread::blockIdx_x();
        let thread = thread::threadIdx_x();
        let lane = thread & 31;
        let warp_in_block = thread / 32;
        let base = chunk * super::RELU2_VALUES_PER_BLOCK;
        let stride = super::RELU2_THREADS_PER_BLOCK;
        let i0 = base + thread;
        let i1 = i0 + stride;
        let i2 = i1 + stride;
        let i3 = i2 + stride;
        let i4 = i3 + stride;
        let i5 = i4 + stride;
        let i6 = i5 + stride;
        let i7 = i6 + stride;
        let local_amax = max_f32(
            max4_f32(
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i0, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i1, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i2, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i3, len),
            ),
            max4_f32(
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i4, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i5, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i6, len),
                relu2_backward_f16_value(pre_activation, d_out, &mut d_pre_activation, i7, len),
            ),
        );

        block_max_store_f32!(
            AMAX,
            d_pre_activation_chunk_amax[chunk],
            local_amax,
            lane,
            warp_in_block
        );
    }

    #[kernel]
    pub fn mlp_block_topk_scores_nvfp4_kernel(
        activation_bytes: &[u8],
        activation_scales: &[u8],
        activation_global_scales: &[f32],
        mut scores: DisjointSlice<f32>,
        token_count: u32,
        feature_count: u32,
    ) {
        static mut SUM: SharedArray<f32, BLOCK_TOPK_SCORE_WARPS> = SharedArray::UNINIT;

        let token_tile = thread::blockIdx_y();
        let feature_tile = thread::blockIdx_x();
        if token_tile >= token_count / BLOCK_TOPK_TILE
            || feature_tile >= feature_count / BLOCK_TOPK_TILE
        {
            return;
        }

        let (thread_id, lane, warp_in_block) = thread_lane_warp();
        let mut pair = thread_id;
        let pairs_per_tile = BLOCK_TOPK_TILE * BLOCK_TOPK_TILE / 2;
        let mut local_sum = 0.0_f32;
        while pair < pairs_per_tile {
            let local = pair * 2;
            let local_row = local / BLOCK_TOPK_TILE;
            let local_col = local - local_row * BLOCK_TOPK_TILE;
            let row = token_tile * BLOCK_TOPK_TILE + local_row;
            let col = feature_tile * BLOCK_TOPK_TILE + local_col;
            let index = (row * feature_count + col) as usize;
            let (lo, hi) = nvfp4_values2(
                activation_bytes,
                activation_scales,
                activation_global_scales[row as usize],
                index,
            );
            local_sum += max_f32(lo, 0.0) + max_f32(hi, 0.0);
            pair += BLOCK_TOPK_SCORE_THREADS;
        }

        let tile_sum = unsafe { block_sum_shared_f32(&mut SUM, local_sum, lane, warp_in_block) };
        if thread_id == 0 {
            let index = token_tile * BLOCK_TOPK_FEATURE_TILES + feature_tile;
            unsafe {
                *scores.get_unchecked_mut(index as usize) = tile_sum;
            }
        }
    }

    #[kernel]
    pub fn mlp_block_topk_masks_kernel(
        scores: &[f32],
        mut masks: DisjointSlice<u64>,
        token_tiles: u32,
        feature_tiles: u32,
        active_feature_tiles: u32,
    ) {
        static mut TOKEN_MASKS: SharedArray<u64, { BLOCK_TOPK_TOKEN_TILES as usize }> =
            SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x();
        let lane = warp::lane_id();
        let warp_in_block = thread_id / 32;
        let mut token_tile = warp_in_block;
        while token_tile < token_tiles {
            let score_base = token_tile * feature_tiles;
            let low_feature = lane;
            let high_feature = lane + 32;
            let low_score = scores[(score_base + low_feature) as usize];
            let high_score = scores[(score_base + high_feature) as usize];
            let mut low_rank = 0_u32;
            let mut high_rank = 0_u32;
            let mut source_lane = 0_u32;
            while source_lane < 32 {
                let other_low_score =
                    warp::shuffle_f32_sync(FULL_WARP_MASK, low_score, source_lane);
                let other_high_score =
                    warp::shuffle_f32_sync(FULL_WARP_MASK, high_score, source_lane);
                low_rank += outranks(other_low_score, source_lane, low_score, low_feature) as u32;
                low_rank +=
                    outranks(other_high_score, source_lane + 32, low_score, low_feature) as u32;
                high_rank +=
                    outranks(other_low_score, source_lane, high_score, high_feature) as u32;
                high_rank +=
                    outranks(other_high_score, source_lane + 32, high_score, high_feature) as u32;
                source_lane += 1;
            }

            let low_mask = warp::ballot_sync(FULL_WARP_MASK, low_rank < active_feature_tiles);
            let high_mask = warp::ballot_sync(FULL_WARP_MASK, high_rank < active_feature_tiles);
            if lane == 0 {
                let mask = low_mask as u64 | ((high_mask as u64) << 32);
                unsafe {
                    TOKEN_MASKS[token_tile as usize] = mask;
                    *masks.get_unchecked_mut(token_tile as usize) = mask;
                }
            }
            token_tile += BLOCK_TOPK_MASK_WARPS;
        }

        thread::sync_threads();

        if thread_id < feature_tiles {
            let mut feature_mask = 0_u64;
            let mut token = 0_u32;
            while token < token_tiles {
                let token_mask = unsafe { TOKEN_MASKS[token as usize] };
                if token_mask & (1_u64 << thread_id) != 0 {
                    feature_mask |= 1_u64 << token;
                }
                token += 1;
            }
            unsafe {
                *masks.get_unchecked_mut((token_tiles + thread_id) as usize) = feature_mask;
            }
        }
    }

    #[inline(always)]
    fn outranks(other_score: f32, other_feature: u32, score: f32, feature: u32) -> bool {
        other_score > score || (other_score == score && other_feature < feature)
    }

    #[inline(always)]
    fn relu2_backward_f16_value(
        pre_activation: &[u16],
        d_out: &[f32],
        d_pre_activation: &mut DisjointSlice<f32>,
        index: u32,
        len: u32,
    ) -> f32 {
        if index >= len {
            return 0.0;
        }

        let pre = cvt_f32_f16(pre_activation[index as usize]);
        let relu = max_f32(pre, 0.0);
        let value = d_out[index as usize] * 2.0 * relu;
        unsafe {
            *d_pre_activation.get_unchecked_mut(index as usize) = value;
        }
        abs_f32(value)
    }
}

pub(super) use module::{LoadedModule, from_module};
