use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::amax::max4_f32;
use crate::block_reduce::{block_max_leader_f32, block_max_store_f32};
use crate::float_ptx::max_f32;
use crate::quartet::quartet_backward_ms_eden_global_scale;
use crate::warp_reduce::thread_lane_warp;

use super::AMAX_WARPS_PER_BLOCK;
use super::input::{
    checked_nvfp4_abs_value, checked_rowwise_abs_value, checked_rowwise_pow2_abs_value,
    nvfp4_value_at, rowwise_value_at, rowwise_value_at_pow2_index,
};
use crate::nvfp4_quant::kernels::row_amax::{tensor_amax_chunk_indices, tensor_chunk_amax8};

// Four-Six rowwise quantization chooses global_scale = row_amax / (256 * 6).
// The group containing the row maximum consequently uses E4M3 scale 256 and
// E2M1 payload 6, so reconstructing that row maximum only requires its stored
// global scale. This is also the exact multiplication performed when the old
// payload scan dequantized the maximum value.
const FOUR_SIX_ROW_AMAX_SCALE: f32 = 256.0 * 6.0;

#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut AMAX_REDUCE: SharedArray<f32, { AMAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;
    static mut ROWWISE_AMAX_REDUCE: SharedArray<f32, 32> = SharedArray::UNINIT;

    #[kernel]
    pub fn rowwise_nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        mut out: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let element_count = rows * cols;
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
            tensor_amax_chunk_indices();

        let local_amax = tensor_chunk_amax8!(
            base,
            element_count,
            [i0, i1, i2, i3, i4, i5, i6, i7],
            rowwise_value_at(bytes, scales, global_scales, cols),
            checked_rowwise_abs_value(bytes, scales, global_scales, cols)
        );

        block_max_store_f32!(AMAX_REDUCE, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn rowwise_nvfp4_chunk_amax_pow2_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scales: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
        row_len_shift: u32,
    ) {
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
            tensor_amax_chunk_indices();

        let local_amax = tensor_chunk_amax8!(
            base,
            element_count,
            [i0, i1, i2, i3, i4, i5, i6, i7],
            rowwise_value_at_pow2_index(bytes, scales, global_scales, row_len_shift),
            checked_rowwise_pow2_abs_value(bytes, scales, global_scales, row_len_shift)
        );

        block_max_store_f32!(AMAX_REDUCE, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn quartet_backward_ms_eden_global_scale_from_rowwise_four_six_kernel(
        scales: &[u8],
        global_scales: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        groups_per_row: u32,
    ) {
        let (thread, lane, warp_in_block) = thread_lane_warp();
        let stride = thread::blockDim_x();
        let mut row = thread;
        let mut local_amax = 0.0;

        while row < row_count {
            local_amax = max_f32(
                local_amax,
                max4_f32(
                    rowwise_four_six_amax(scales, global_scales, row, row_count, groups_per_row),
                    rowwise_four_six_amax(
                        scales,
                        global_scales,
                        row + stride,
                        row_count,
                        groups_per_row,
                    ),
                    rowwise_four_six_amax(
                        scales,
                        global_scales,
                        row + stride * 2,
                        row_count,
                        groups_per_row,
                    ),
                    rowwise_four_six_amax(
                        scales,
                        global_scales,
                        row + stride * 3,
                        row_count,
                        groups_per_row,
                    ),
                ),
            );
            row += stride * 4;
        }

        if let Some(amax) = unsafe {
            block_max_leader_f32(&mut ROWWISE_AMAX_REDUCE, local_amax, lane, warp_in_block)
        } {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) =
                    quartet_backward_ms_eden_global_scale(amax);
            }
        }
    }

    #[kernel]
    pub fn quartet_backward_ms_eden_global_scale_from_four_six_kernel(
        scales: &[u8],
        global_scale: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
        group_count: u32,
    ) {
        let (thread, lane, warp_in_block) = thread_lane_warp();
        let source_global_scale = global_scale[0];

        if source_global_scale != 1.0 {
            if thread == 0 {
                unsafe {
                    *out_global_scale.get_unchecked_mut(0) = quartet_backward_ms_eden_global_scale(
                        source_global_scale * FOUR_SIX_ROW_AMAX_SCALE,
                    );
                }
            }
            return;
        }

        // As in the rowwise path, 1.0 is also the safety scale for an all-zero
        // tensor. Inspect scale bytes only for that ambiguous sentinel.
        let mut group = thread;
        let mut local_amax = 0.0;
        let stride = thread::blockDim_x();
        while group < group_count {
            if crate::nvfp4::csf::scale_at(scales, group as usize) != 0 {
                local_amax = FOUR_SIX_ROW_AMAX_SCALE;
            }
            group += stride;
        }

        if let Some(amax) =
            unsafe { block_max_leader_f32(&mut AMAX_REDUCE, local_amax, lane, warp_in_block) }
        {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) =
                    quartet_backward_ms_eden_global_scale(amax);
            }
        }
    }

    #[inline(always)]
    fn rowwise_four_six_amax(
        scales: &[u8],
        global_scales: &[f32],
        row: u32,
        row_count: u32,
        groups_per_row: u32,
    ) -> f32 {
        if row >= row_count {
            return 0.0;
        }

        let global_scale = global_scales[row as usize];
        if global_scale != 1.0 {
            return global_scale * FOUR_SIX_ROW_AMAX_SCALE;
        }

        // A zero row stores the nonzero safety scale 1.0 while all of its
        // block scales remain zero. Only that ambiguous sentinel needs to
        // inspect scale bytes; ordinary rows take the single-load path.
        let group_base = row * groups_per_row;
        let mut group = 0;
        while group < groups_per_row {
            if unsafe {
                crate::nvfp4::csf::matrix_scale_at_data(
                    scales.as_ptr(),
                    global_scales.len() as u32,
                    (groups_per_row * 16).div_ceil(128) * 128,
                    row,
                    group,
                )
            } != 0
            {
                return FOUR_SIX_ROW_AMAX_SCALE;
            }
            group += 1;
        }
        0.0
    }

    #[kernel]
    pub fn nvfp4_chunk_amax_kernel(
        bytes: &[u8],
        scales: &[u8],
        global_scale: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
    ) {
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
            tensor_amax_chunk_indices();

        let local_amax = tensor_chunk_amax8!(
            base,
            element_count,
            [i0, i1, i2, i3, i4, i5, i6, i7],
            nvfp4_value_at(bytes, scales, global_scale),
            checked_nvfp4_abs_value(bytes, scales, global_scale)
        );

        block_max_store_f32!(AMAX_REDUCE, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn quartet_backward_ms_eden_global_scale_from_chunks_kernel(
        chunk_amax: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
        chunk_count: u32,
    ) {
        let (thread, lane, warp_in_block) = thread_lane_warp();
        let mut chunk = thread;
        let mut local_amax = 0.0;
        let stride = thread::blockDim_x();

        while chunk < chunk_count {
            local_amax = max_f32(
                local_amax,
                max4_f32(
                    chunk_amax_or_zero(chunk_amax, chunk, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride * 2, chunk_count),
                    chunk_amax_or_zero(chunk_amax, chunk + stride * 3, chunk_count),
                ),
            );
            chunk += stride * 4;
        }

        if let Some(amax) =
            unsafe { block_max_leader_f32(&mut AMAX_REDUCE, local_amax, lane, warp_in_block) }
        {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) =
                    quartet_backward_ms_eden_global_scale(amax);
            }
        }
    }

    #[inline(always)]
    fn chunk_amax_or_zero(chunk_amax: &[f32], chunk: u32, chunk_count: u32) -> f32 {
        if chunk < chunk_count {
            chunk_amax[chunk as usize]
        } else {
            0.0
        }
    }
}
