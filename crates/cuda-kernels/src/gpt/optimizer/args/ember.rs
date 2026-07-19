use cuda_core::{CudaStream, DeviceBuffer};

pub struct EmberUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub z_master: &'a mut DeviceBuffer<f32>,
    pub x_master: &'a mut DeviceBuffer<f32>,
    pub grad: &'a DeviceBuffer<f32>,
    pub row_second_moment: &'a mut DeviceBuffer<f32>,
    pub column_second_moment: &'a mut DeviceBuffer<f32>,
    pub column_partials: &'a mut DeviceBuffer<f32>,
    pub normalizer: &'a mut DeviceBuffer<f32>,
    pub rows: u32,
    pub cols: u32,
    pub grad_scale: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta2: f32,
    pub beta2_correction: f32,
    pub eps: f32,
    pub average_coefficient: f32,
}
