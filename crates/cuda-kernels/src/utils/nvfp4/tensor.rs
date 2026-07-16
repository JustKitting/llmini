use cuda_core::DeviceBuffer;

use crate::nvfp4_cast::{e2m1_value, e2m1_values, e4m3_value};

#[derive(Clone, Copy)]
pub struct Nvfp4DeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scale: &'a DeviceBuffer<f32>,
}

impl<'a> Nvfp4DeviceTensor<'a> {
    pub fn new(
        bytes: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
        global_scale: &'a DeviceBuffer<f32>,
    ) -> Self {
        Self {
            bytes,
            scales,
            global_scale,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Nvfp4RowwiseDeviceTensor<'a> {
    pub bytes: &'a DeviceBuffer<u8>,
    pub scales: &'a DeviceBuffer<u8>,
    pub global_scales: &'a DeviceBuffer<f32>,
}

impl<'a> Nvfp4RowwiseDeviceTensor<'a> {
    pub fn new(
        bytes: &'a DeviceBuffer<u8>,
        scales: &'a DeviceBuffer<u8>,
        global_scales: &'a DeviceBuffer<f32>,
    ) -> Self {
        Self {
            bytes,
            scales,
            global_scales,
        }
    }
}

#[inline(always)]
pub fn nvfp4_value(bytes: &[u8], scales: &[u8], global_scale: f32, index: usize) -> f32 {
    let byte = bytes[index / 2];
    let payload = if index & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };

    e2m1_value(payload) * e4m3_value(scales[index / 16] as u16) * global_scale
}

#[inline(always)]
pub fn nvfp4_values2(
    bytes: &[u8],
    scales: &[u8],
    global_scale: f32,
    even_index: usize,
) -> (f32, f32) {
    let (lo, hi) = e2m1_values(bytes[even_index / 2]);
    let scale = e4m3_value(scales[even_index / 16] as u16);
    (lo * scale * global_scale, hi * scale * global_scale)
}

#[inline(always)]
pub fn nvfp4_rowwise_value(
    bytes: &[u8],
    scales: &[u8],
    global_scales: &[f32],
    row_len: usize,
    row: usize,
    col: usize,
) -> f32 {
    let index = row * row_len + col;
    nvfp4_value(bytes, scales, global_scales[row], index)
}
