use cuda_core::{CudaStream, DeviceBuffer};

pub struct GradientClipArgs<'a> {
    pub stream: &'a CudaStream,
    pub chunk_ptrs: &'a DeviceBuffer<u64>,
    pub chunk_lens: &'a DeviceBuffer<u32>,
    pub chunk_sums: &'a mut DeviceBuffer<f32>,
    pub scale: &'a mut DeviceBuffer<f32>,
    pub norm: &'a mut DeviceBuffer<f32>,
    pub chunk_count: u32,
    pub max_norm: f32,
    pub apply: bool,
}
