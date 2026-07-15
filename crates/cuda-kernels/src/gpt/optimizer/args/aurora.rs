use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AuroraSlotDescriptor {
    pub grad: u64,
    pub momentum: u64,
    pub z_master: u64,
    pub x_master: u64,
    pub bytes: u64,
    pub scales: u64,
    pub global_scale: u64,
    pub rows: u32,
    pub cols: u32,
    pub learning_rate_multiplier: f32,
}

unsafe impl DeviceCopy for AuroraSlotDescriptor {}

pub struct AuroraMegaUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<AuroraSlotDescriptor>,
    pub oriented: &'a mut DeviceBuffer<f32>,
    pub polar_next: &'a mut DeviceBuffer<f32>,
    pub polar_x: &'a mut DeviceBuffer<f32>,
    pub polar_gram: &'a mut DeviceBuffer<f32>,
    pub polar_ax: &'a mut DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub slot_count: u32,
    pub max_len: u32,
    pub max_ax_len: u32,
    pub max_dim: u32,
    pub mu: f32,
    pub grad_scale: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
    pub iterations: u32,
}

pub struct AuroraTmaPrepareArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<AuroraSlotDescriptor>,
    pub oriented: &'a mut DeviceBuffer<f32>,
    pub polar_x: &'a mut DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub slot_index: u32,
    pub mu: f32,
    pub grad_scale: f32,
}

pub struct AuroraTmaFinishArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<AuroraSlotDescriptor>,
    pub polar_update: &'a DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub slot_index: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
}
