use cuda_core::{CudaStream, DeviceBuffer};

pub struct ScheduleFreeMaterializeArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
    pub z_master: &'a DeviceBuffer<f32>,
    pub x_master: &'a DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub chunk_amax: &'a mut DeviceBuffer<f32>,
    pub len: u32,
    pub beta: f32,
}

pub struct ScheduleFreeMaterializePrecomputedArgs<'a> {
    pub stream: &'a CudaStream,
    pub bytes: &'a mut DeviceBuffer<u8>,
    pub scales: &'a mut DeviceBuffer<u8>,
    pub global_scale: &'a mut DeviceBuffer<f32>,
    pub z_master: &'a DeviceBuffer<f32>,
    pub x_master: &'a DeviceBuffer<f32>,
    pub amax: &'a DeviceBuffer<f32>,
    pub len: u32,
    pub beta: f32,
}
