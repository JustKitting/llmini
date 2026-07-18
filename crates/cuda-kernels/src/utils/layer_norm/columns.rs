use cuda_device::DisjointSlice;

use crate::f16_tc_matmul::convert::{cvt_f32_f16, cvt_rn_f16_f32};
use crate::float_ptx::fma_f32;
use crate::nvfp4::nvfp4_value;

#[inline(always)]
pub fn f32_column(values: &[f32], row_base: usize, col: u32, row_len: u32) -> f32 {
    if col < row_len {
        values[row_base + col as usize]
    } else {
        0.0
    }
}

#[inline(always)]
pub fn f16_column(values: &[u16], row_base: usize, col: u32, row_len: u32) -> f32 {
    if col < row_len {
        cvt_f32_f16(values[row_base + col as usize])
    } else {
        0.0
    }
}

#[inline(always)]
pub fn nvfp4_column(
    bytes: &[u8],
    scales: &[u8],
    global_scale: f32,
    row_base: usize,
    col: u32,
    row_len: u32,
) -> f32 {
    if col < row_len {
        nvfp4_value(bytes, scales, global_scale, row_base + col as usize)
    } else {
        0.0
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn nvfp4_affine_normalized_column(
    weight_bytes: &[u8],
    weight_scales: &[u8],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    col: u32,
    row_len: u32,
    centered: f32,
    inv_std: f32,
    weight_global_scale: f32,
    bias_global_scale: f32,
) -> f32 {
    if col < row_len {
        let weight = nvfp4_value(
            weight_bytes,
            weight_scales,
            weight_global_scale,
            col as usize,
        );
        let bias = nvfp4_value(bias_bytes, bias_scales, bias_global_scale, col as usize);
        fma_f32(centered * inv_std, weight, bias)
    } else {
        0.0
    }
}

#[inline(always)]
pub fn store_f16_column(
    values: &mut DisjointSlice<'_, u16>,
    row_base: usize,
    col: u32,
    row_len: u32,
    value: f32,
) {
    if col < row_len {
        unsafe {
            *values.get_unchecked_mut(row_base + col as usize) = cvt_rn_f16_f32(value);
        }
    }
}
