use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::amax::max4_f32;
use crate::block_reduce::block_max_store_f32;
use crate::float_ptx::{abs_f32, max_f32};
use crate::warp_reduce::thread_lane_warp;

use super::super::config::WARPS_PER_BLOCK;

pub(crate) const TENSOR_AMAX_VALUES_PER_BLOCK: u32 = 2048;

macro_rules! tensor_chunk_amax8 {
    ($base:expr, $count:expr, [$i0:expr, $i1:expr, $i2:expr, $i3:expr, $i4:expr, $i5:expr, $i6:expr, $i7:expr], $value:ident($($value_arg:expr),+), $checked:ident($($checked_arg:expr),+)) => {{
        if $base + $crate::nvfp4_quant::kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK <= $count {
            $crate::float_ptx::max_f32(
                $crate::amax::amax4_f32($value($($value_arg),+, $i0), $value($($value_arg),+, $i1), $value($($value_arg),+, $i2), $value($($value_arg),+, $i3)),
                $crate::amax::amax4_f32($value($($value_arg),+, $i4), $value($($value_arg),+, $i5), $value($($value_arg),+, $i6), $value($($value_arg),+, $i7)),
            )
        } else {
            $crate::float_ptx::max_f32(
                $crate::amax::max4_f32($checked($($checked_arg),+, $i0, $count), $checked($($checked_arg),+, $i1, $count), $checked($($checked_arg),+, $i2, $count), $checked($($checked_arg),+, $i3, $count)),
                $crate::amax::max4_f32($checked($($checked_arg),+, $i4, $count), $checked($($checked_arg),+, $i5, $count), $checked($($checked_arg),+, $i6, $count), $checked($($checked_arg),+, $i7, $count)),
            )
        }
    }};
}
pub(crate) use tensor_chunk_amax8;

#[inline(always)]
pub(crate) fn tensor_amax_chunk_indices()
-> (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let chunk = thread::blockIdx_x();
    let (thread, lane, warp_in_block) = thread_lane_warp();
    let base = chunk * TENSOR_AMAX_VALUES_PER_BLOCK;
    let stride = thread::blockDim_x();
    let i0 = base + thread;
    let i1 = i0 + stride;
    let i2 = i1 + stride;
    let i3 = i2 + stride;
    let i4 = i3 + stride;
    let i5 = i4 + stride;
    let i6 = i5 + stride;
    let i7 = i6 + stride;
    (
        chunk,
        lane,
        warp_in_block,
        base,
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
    )
}

#[cuda_module]
pub(crate) mod module {
    use super::*;

    static mut ROW_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
    static mut TENSOR_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    #[kernel]
    pub fn row_amax_f32_kernel(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        row_count: u32,
        row_len: u32,
    ) {
        let row = thread::blockIdx_x();
        let (thread, lane, warp_in_block) = thread_lane_warp();

        if row < row_count {
            let row_base = row as usize * row_len as usize;
            let mut col = thread;
            let mut local_amax = 0.0;

            while col < row_len {
                local_amax = max_f32(local_amax, abs_f32(x[row_base + col as usize]));
                col += thread::blockDim_x();
            }

            block_max_store_f32!(ROW_AMAX, out[row], local_amax, lane, warp_in_block);
        }
    }

    #[kernel]
    pub fn tensor_chunk_amax_f32_kernel(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        element_count: u32,
    ) {
        let (chunk, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
            tensor_amax_chunk_indices();

        let local_amax = tensor_chunk_amax8!(
            base,
            element_count,
            [i0, i1, i2, i3, i4, i5, i6, i7],
            abs_f32_at(x),
            checked_abs_f32(x)
        );

        block_max_store_f32!(TENSOR_AMAX, out[chunk], local_amax, lane, warp_in_block);
    }

    #[kernel]
    pub fn tensor_amax_from_chunks_f32_kernel(
        chunk_amax: &[f32],
        mut out: DisjointSlice<f32>,
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

        block_max_store_f32!(TENSOR_AMAX, out[0], local_amax, lane, warp_in_block);
    }

    #[inline(always)]
    fn chunk_amax_or_zero(chunk_amax: &[f32], chunk: u32, chunk_count: u32) -> f32 {
        if chunk < chunk_count {
            chunk_amax[chunk as usize]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn abs_f32_at(x: &[f32], index: u32) -> f32 {
        abs_f32(x[index as usize])
    }

    #[inline(always)]
    fn checked_abs_f32(x: &[f32], index: u32, element_count: u32) -> f32 {
        if index < element_count {
            abs_f32(x[index as usize])
        } else {
            0.0
        }
    }
}
