use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MuonSlotDescriptor {
    pub grad: u64,
    pub momentum: u64,
    pub variance: u64,
    pub second_momentum: u64,
    pub z_master: u64,
    pub x_master: u64,
    pub schedule_amax: u64,
    pub bytes: u64,
    pub scales: u64,
    pub global_scale: u64,
    pub rows: u32,
    pub cols: u32,
    pub learning_rate_multiplier: f32,
    pub qk_clip_factor_offset: u32,
    pub qk_clip_head_dim: u32,
}

unsafe impl DeviceCopy for MuonSlotDescriptor {}

pub struct MuonMegaUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<MuonSlotDescriptor>,
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

pub struct MuonTmaPrepareArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<MuonSlotDescriptor>,
    pub oriented: &'a mut DeviceBuffer<f32>,
    pub polar_x: &'a mut DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub polar_x_chunk_amax: &'a mut DeviceBuffer<f32>,
    pub slot_index: u32,
    pub matrix_len: u32,
    pub mu: f32,
    pub grad_scale: f32,
    pub nesterov: u32,
    pub variance_adaptive: u32,
    pub bias_correction_inv: f32,
}

pub struct MuonTmaFinishArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<MuonSlotDescriptor>,
    pub polar_update: &'a DeviceBuffer<f32>,
    pub polar_bound_amax: &'a DeviceBuffer<f32>,
    pub polar_chunks: &'a mut DeviceBuffer<f32>,
    pub normuon_factors: &'a mut DeviceBuffer<f32>,
    pub normuon_chunks: &'a mut DeviceBuffer<f32>,
    pub qk_clip_factors: &'a DeviceBuffer<f32>,
    pub slot_index: u32,
    pub matrix_len: u32,
    pub polar_cols: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
    pub schedule_beta: f32,
    pub apply_polar_sqrt_bound: u32,
}

pub struct MuonTmaHyperballFinishArgs<'a> {
    pub finish: MuonTmaFinishArgs<'a>,
    pub hyperball_chunks: &'a mut DeviceBuffer<f32>,
    pub use_schedule_free: bool,
}

pub struct MuonTmaSignUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub slots: &'a DeviceBuffer<MuonSlotDescriptor>,
    pub update_chunks: &'a mut DeviceBuffer<f32>,
    pub qk_clip_factors: &'a DeviceBuffer<f32>,
    pub slot_index: u32,
    pub matrix_len: u32,
    pub mu: f32,
    pub grad_scale: f32,
    pub variance_adaptive: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub average_coefficient: f32,
    pub schedule_beta: f32,
}
