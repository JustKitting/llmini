use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::HostPtrs;

pub(super) struct MuonPaddingBuffers {
    grad: DeviceBuffer<f32>,
    momentum: DeviceBuffer<f32>,
    variance: DeviceBuffer<u16>,
    second_momentum: DeviceBuffer<f32>,
    z_master: DeviceBuffer<f32>,
    x_master: DeviceBuffer<f32>,
    schedule_amax: DeviceBuffer<f32>,
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl MuonPaddingBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            grad: DeviceBuffer::zeroed(stream, 16)?,
            momentum: DeviceBuffer::zeroed(stream, 16)?,
            variance: DeviceBuffer::zeroed(stream, 16)?,
            second_momentum: DeviceBuffer::zeroed(stream, 1)?,
            z_master: DeviceBuffer::zeroed(stream, 16)?,
            x_master: DeviceBuffer::zeroed(stream, 16)?,
            schedule_amax: DeviceBuffer::zeroed(stream, 1)?,
            bytes: DeviceBuffer::zeroed(stream, 8)?,
            scales: DeviceBuffer::zeroed(stream, 1)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub fn ptrs(&self) -> HostPtrs {
        HostPtrs {
            grad: self.grad.cu_deviceptr(),
            momentum: self.momentum.cu_deviceptr(),
            variance: self.variance.cu_deviceptr(),
            second_momentum: self.second_momentum.cu_deviceptr(),
            z_master: self.z_master.cu_deviceptr(),
            x_master: self.x_master.cu_deviceptr(),
            schedule_amax: self.schedule_amax.cu_deviceptr(),
            bytes: self.bytes.cu_deviceptr(),
            scales: self.scales.cu_deviceptr(),
            global_scale: self.global_scale.cu_deviceptr(),
            rows: 0,
            cols: 0,
            learning_rate_multiplier: 1.0,
            qk_clip_factor_offset: u32::MAX,
            qk_clip_head_dim: 0,
            symexp_exponential_z: 0,
            symexp_exponential_x: 0,
            symexp_exponential_grad: 0,
            symexp_linear_z: 0,
            symexp_linear_x: 0,
            symexp_linear_grad: 0,
            symexp_curvature_z: 0,
            symexp_curvature_x: 0,
            symexp_curvature_grad: 0,
        }
    }
}
