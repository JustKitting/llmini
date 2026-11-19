use cuda_core::{CudaStream, DeviceBuffer};

use super::MsEdenDeviceScaleQuantArgs;

pub struct QuartetBackwardMsEdenQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,
    pub out_global_scales: &'out mut DeviceBuffer<f32>,
    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

pub struct QuartetBackwardMsEdenDeviceScaleQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub x: &'a DeviceBuffer<f32>,
    pub out_fp4: &'out mut DeviceBuffer<u8>,
    pub out_scales: &'out mut DeviceBuffer<u8>,

    pub out_chunk_amax: &'out mut DeviceBuffer<f32>,
    pub out_global_scale: &'out mut DeviceBuffer<f32>,
    pub row_count: u32,
    pub src_row_len: u32,
    pub dst_row_len: u32,
    pub sign_seed: u32,
    pub scale_seed: u32,
}

impl QuartetBackwardMsEdenDeviceScaleQuantArgs<'_, '_> {
    pub(crate) fn device_scale_args(
        &mut self,
        scale_override: f32,
    ) -> MsEdenDeviceScaleQuantArgs<'_, '_> {
        MsEdenDeviceScaleQuantArgs {
            stream: self.stream,
            x: self.x,
            out_fp4: &mut *self.out_fp4,
            out_scales: &mut *self.out_scales,

            out_chunk_amax: &mut *self.out_chunk_amax,
            global_scale: &*self.out_global_scale,
            row_count: self.row_count,
            src_row_len: self.src_row_len,
            dst_row_len: self.dst_row_len,
            scale_override,
            sign_seed: self.sign_seed,
            scale_seed: self.scale_seed,
        }
    }
}
