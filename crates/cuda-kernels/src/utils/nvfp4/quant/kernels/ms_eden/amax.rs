use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::amax::max4_f32;
use crate::block_reduce::{block_max_leader_f32, block_max_store_f32};
use crate::float_ptx::max_f32;
use crate::quartet::quartet_backward_ms_eden_global_scale;
use crate::warp_reduce::thread_lane_warp;

use super::AMAX_WARPS_PER_BLOCK;
use super::input::{
    checked_nvfp4_abs_value, checked_rowwise_abs_value, nvfp4_value_at, rowwise_value_at,
};
use crate::nvfp4_quant::kernels::row_amax::{tensor_amax_chunk_indices, tensor_chunk_amax8};

#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut AMAX_REDUCE: SharedArray<f32, { AMAX_WARPS_PER_BLOCK as usize }> =
        SharedArray::UNINIT;

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
