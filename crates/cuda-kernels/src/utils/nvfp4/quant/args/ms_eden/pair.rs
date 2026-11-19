use cuda_core::{CudaStream, DeviceBuffer};

pub struct MsEdenPairDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub transpose_out_fp4: &'out mut DeviceBuffer<u8>,
    pub transpose_out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub transpose_dst_row_len: u32,
    pub scale_override: f32,
    pub sign_seed: u32,
    pub scale_seed: u32,
    pub transpose_scale_seed: u32,
    pub precomputed_chunk_count: Option<u32>,
}
