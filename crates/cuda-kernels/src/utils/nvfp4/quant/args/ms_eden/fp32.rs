use cuda_core::{CudaStream, DeviceBuffer};

pub struct MsEdenQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub global_scale: f32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct MsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub global_scale: &'a DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct MsEdenTransposeDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub global_scale: &'a DeviceBuffer<f32>,
    pub source_rows: u32,
    pub source_cols: u32,
    pub dst_row_len: u32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}
