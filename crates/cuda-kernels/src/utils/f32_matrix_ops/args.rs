use cuda_core::{CudaStream, DeviceBuffer};

pub struct F32Linear2Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
    pub a_scale: f32,
    pub b_scale: f32,
}

pub struct F32Linear3Args<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub c_out: &'out mut DeviceBuffer<f32>,
    pub len: u32,
    pub a_scale: f32,
    pub b_scale: f32,
    pub c_scale: f32,
}

pub struct F32Linear3SqrtBoundArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub c_out: &'out mut DeviceBuffer<f32>,
    pub bound_amax: &'a DeviceBuffer<f32>,
    pub len: u32,
    pub a_scale: f32,
    pub b_scale: f32,
    pub c_scale: f32,
}

pub struct F32Linear3SqrtBoundAmaxArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub c_out: &'out mut DeviceBuffer<f32>,
    pub bound_amax: &'a DeviceBuffer<f32>,
    pub chunk_amax: &'out mut DeviceBuffer<f32>,
    pub len: u32,
    pub a_scale: f32,
    pub b_scale: f32,
    pub c_scale: f32,
}

pub struct F32Linear3SqrtBoundRowSumsqArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub a: &'a DeviceBuffer<f32>,
    pub b: &'a DeviceBuffer<f32>,
    pub c_out: &'out mut DeviceBuffer<f32>,
    pub bound_amax: &'a DeviceBuffer<f32>,
    pub row_sumsq: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
    pub a_scale: f32,
    pub b_scale: f32,
    pub c_scale: f32,
}

pub struct F32AddScaledIdentityArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub src: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<f32>,
    pub dim: u32,
    pub scale: f32,
}

pub struct F32ScaleInPlaceByAmaxArgs<'a> {
    pub stream: &'a CudaStream,
    pub x: &'a mut DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub len: u32,
}
