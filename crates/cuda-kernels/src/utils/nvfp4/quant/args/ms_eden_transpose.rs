use cuda_core::{CudaStream, DeviceBuffer};

use crate::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};

pub struct RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4RowwiseDeviceTensor<'a>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub source_rows: u32,
    pub source_cols: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct Nvfp4TransposeMsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: Nvfp4DeviceTensor<'a>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub source_rows: u32,
    pub source_cols: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}
