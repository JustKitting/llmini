use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::amax::max4_f32;
use crate::block_reduce::block_max_store_f32;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{abs_f32, max_f32};
use crate::mma::{
    Nvfp4ProjectionParams, dispatch_projection_cta_tiles, nvfp4_projection_cta_kernel_body,
    nvfp4_projection_cta_kernel_body_at_aligned_row_pair, nvfp4_projection_cta_relu2_kernel_body,
    nvfp4_projection_cta_relu2_kernel_body_at_aligned_row_pair,
};

pub(super) const RELU2_THREADS_PER_BLOCK: u32 = 256;
pub(super) const RELU2_VALUES_PER_BLOCK: u32 = 8 * RELU2_THREADS_PER_BLOCK;
const RELU2_WARPS_PER_BLOCK: usize = (RELU2_THREADS_PER_BLOCK / 32) as usize;

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
