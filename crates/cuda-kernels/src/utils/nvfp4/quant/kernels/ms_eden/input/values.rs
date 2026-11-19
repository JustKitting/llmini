use crate::float_ptx::abs_f32;
use crate::nvfp4::{nvfp4_rowwise_value, nvfp4_value};
use crate::nvfp4_cast::{e2m1_value, e4m3_value};

#[inline(always)]
pub(super) fn nvfp4_value_at(bytes: &[u8], scales: &[u8], global_scale: &[f32], index: u32) -> f32 {
    nvfp4_value(bytes, scales, global_scale[0], index as usize)
}

#[inline(always)]
pub(super) fn checked_nvfp4_abs_value(
    bytes: &[u8],
    scales: &[u8],
    global_scale: &[f32],
    index: u32,
    element_count: u32,
) -> f32 {
    if index < element_count {
        abs_f32(nvfp4_value_at(bytes, scales, global_scale, index))
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn rowwise_value_at(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    cols: u32,
    index: u32,
) -> f32 {
    let row = index / cols;
    let col = index - row * cols;
    nvfp4_rowwise_value(
        bytes,
        scales,
        global_scales,
        cols as usize,
        row as usize,
        col as usize,
    )
}

#[inline(always)]
pub(super) fn rowwise_value_at_pow2_index(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    row_len_shift: u32,
    index: u32,
) -> f32 {
    let byte = bytes[(index >> 1) as usize];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };

    e2m1_value(payload)
        * e4m3_value(unsafe {
            crate::nvfp4::csf::matrix_scale_at_data(
                scales.as_ptr(),
                global_scales.len() as u32,
                (1u32 << row_len_shift).div_ceil(128) * 128,
                index >> row_len_shift,
                (index & ((1u32 << row_len_shift) - 1)) / 16,
            )
        } as u16)
        * global_scales[(index >> row_len_shift) as usize]
}

#[inline(always)]
pub(super) fn checked_rowwise_abs_value(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    cols: u32,
    index: u32,
    element_count: u32,
) -> f32 {
    if index < element_count {
        abs_f32(rowwise_value_at(bytes, scales, global_scales, cols, index))
    } else {
        0.0
    }
}

#[inline(always)]
pub(super) fn checked_rowwise_pow2_abs_value(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    row_len_shift: u32,
    index: u32,
    element_count: u32,
) -> f32 {
    if index < element_count {
        abs_f32(rowwise_value_at_pow2_index(
            bytes,
            scales,
            global_scales,
            row_len_shift,
            index,
        ))
    } else {
        0.0
    }
}
