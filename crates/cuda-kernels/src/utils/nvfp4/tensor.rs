use cuda_core::DeviceBuffer;

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
pub fn nvfp4_value(bytes: &[u8], scales: &[u8], global: f32, index: usize) -> f32 {
    let pair = crate::nvfp4_cast::scaled_e2m1_half2(
        bytes[index / 2],
        super::csf::scale_at(scales, index / 16),
    );
    let h = if index & 1 == 0 {
        pair as u16
    } else {
        (pair >> 16) as u16
    };
    crate::nvfp4_cast::f16_value(h) * global
}

#[inline(always)]
pub fn nvfp4_vector_value(bytes: &[u8], scales: &[u8], global: f32, index: usize) -> f32 {
    let pair = crate::nvfp4_cast::scaled_e2m1_half2(
        bytes[index / 2],
        super::csf::vector_scale_at(scales, index / 16),
    );
    let h = if index & 1 == 0 {
        pair as u16
    } else {
        (pair >> 16) as u16
    };
    crate::nvfp4_cast::f16_value(h) * global
}

#[inline(always)]
pub fn nvfp4_values2(bytes: &[u8], scales: &[u8], global: f32, index: usize) -> (f32, f32) {
    let pair = crate::nvfp4_cast::scaled_e2m1_half2(
        bytes[index / 2],
        super::csf::scale_at(scales, index / 16),
    );
    (
        crate::nvfp4_cast::f16_value(pair as u16) * global,
        crate::nvfp4_cast::f16_value((pair >> 16) as u16) * global,
    )
}

#[inline(always)]
pub fn nvfp4_vector_values2(bytes: &[u8], scales: &[u8], global: f32, index: usize) -> (f32, f32) {
    let pair = crate::nvfp4_cast::scaled_e2m1_half2(
        bytes[index / 2],
        super::csf::vector_scale_at(scales, index / 16),
    );
    (
        crate::nvfp4_cast::f16_value(pair as u16) * global,
        crate::nvfp4_cast::f16_value((pair >> 16) as u16) * global,
    )
}

#[inline(always)]
pub fn nvfp4_rowwise_value(
    bytes: &[u8],
    scales: &[u8],
    globals: &[f32],
    row_len: usize,
    row: usize,
    col: usize,
) -> f32 {
    let index = row * row_len + col;
    let pair = crate::nvfp4_cast::scaled_e2m1_half2(bytes[index / 2], unsafe {
        super::csf::matrix_scale_at_data(
            scales.as_ptr(),
            globals.len() as u32,
            row_len.div_ceil(128) as u32 * 128,
            row as u32,
            (col / 16) as u32,
        )
    });
    let h = if index & 1 == 0 {
        pair as u16
    } else {
        (pair >> 16) as u16
    };
    crate::nvfp4_cast::f16_value(h) * globals[row]
}

#[inline(always)]
pub fn nvfp4_rowwise_halves8(
    bytes: &[u8],
    scales: &[u8],
    row_len: u32,
    row: u32,
    col: u32,
) -> u128 {
    let index = (row * row_len + col) as usize;
    let scale = unsafe {
        super::csf::matrix_scale_at_data(
            scales.as_ptr(),
            0,
            row_len.div_ceil(128) * 128,
            row,
            col / 16,
        )
    };
    let packed: u32;
    unsafe {
        cuda_device::ptx_asm!("ld.global.nc.u32 %0,[%1];",out("=r")packed,in("l")bytes.as_ptr().add(index/2)as u64,clobber("memory"));
    }
    crate::nvfp4_cast::scaled_e2m1_half8(packed, scale)
}

/// Four adjacent values share their stored FP8 scale and packed conversion.
#[inline(always)]
pub fn nvfp4_vector_values4(bytes: &[u8], scales: &[u8], global: f32, index: usize) -> [f32; 4] {
    let bits: u16;
    unsafe {
        cuda_device::ptx_asm!("ld.global.nc.u16 %0,[%1];",out("=h")bits,in("l")bytes.as_ptr().add(index/2)as u64,options(register_only));
    }
    let halves =
        crate::nvfp4_cast::scaled_e2m1_half4(bits, super::csf::vector_scale_at(scales, index / 16));
    [
        crate::nvfp4_cast::f16_value(halves as u16) * global,
        crate::nvfp4_cast::f16_value((halves >> 16) as u16) * global,
        crate::nvfp4_cast::f16_value((halves >> 32) as u16) * global,
        crate::nvfp4_cast::f16_value((halves >> 48) as u16) * global,
    ]
}
