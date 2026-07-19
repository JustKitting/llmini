use cuda_core::{CudaStream, DeviceBuffer};

#[derive(Clone, Copy)]
pub struct SymExpLinScaleRefs<'a> {
    pub exponential_z: &'a DeviceBuffer<f32>,
    pub exponential_x: &'a DeviceBuffer<f32>,
    pub linear_z: &'a DeviceBuffer<f32>,
    pub linear_x: &'a DeviceBuffer<f32>,
    pub curvature_z: &'a DeviceBuffer<f32>,
    pub curvature_x: &'a DeviceBuffer<f32>,
}

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
    pub symexp_lin_beta: f32,
    pub symexp_lin_scales: Option<SymExpLinScaleRefs<'a>>,
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
    pub symexp_lin_beta: f32,
    pub symexp_lin_scales: Option<SymExpLinScaleRefs<'a>>,
}
