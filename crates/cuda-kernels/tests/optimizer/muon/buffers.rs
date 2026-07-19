use std::error::Error;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use rust_kernels_cuda::optimizer::{
    MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES, MuonSlotDescriptor,
};

pub const SLOT_COUNT: usize = MUON_MATRIX_PHASES;

pub struct Slots {
    pub grads: Vec<DeviceBuffer<f32>>,
    pub momentums: Vec<DeviceBuffer<f32>>,
    pub z_masters: Vec<DeviceBuffer<f32>>,
    pub x_masters: Vec<DeviceBuffer<f32>>,
    pub schedule_amaxes: Vec<DeviceBuffer<f32>>,
    pub bytes: Vec<DeviceBuffer<u8>>,
    pub scales: Vec<DeviceBuffer<u8>>,
    pub global_scales: Vec<DeviceBuffer<f32>>,
}

pub struct Scratch {
    pub oriented: DeviceBuffer<f32>,
    pub polar_next: DeviceBuffer<f32>,
    pub polar_x: DeviceBuffer<f32>,
    pub polar_gram: DeviceBuffer<f32>,
    pub polar_ax: DeviceBuffer<f32>,
    pub polar_chunks: DeviceBuffer<f32>,
}

impl Slots {
    pub fn new(stream: &CudaStream, grad: &[f32]) -> Result<Self, Box<dyn Error>> {
        let len = grad.len();
        Ok(Self {
            grads: slot_buffers(stream, grad)?,
            momentums: zero_slot_buffers::<f32>(stream, len)?,
            z_masters: fill_slot_buffers(stream, 1.0, len)?,
            x_masters: fill_slot_buffers(stream, 1.0, len)?,
            schedule_amaxes: zero_slot_buffers::<f32>(stream, 1)?,
            bytes: zero_slot_buffers::<u8>(stream, len / 2)?,
            scales: zero_slot_buffers::<u8>(stream, len / 16)?,
            global_scales: zero_slot_buffers::<f32>(stream, 1)?,
        })
    }

    pub fn with_repeated_grad(
        stream: &CudaStream,
        grad: f32,
        len: usize,
    ) -> Result<Self, Box<dyn Error>> {
        Self::new(stream, &vec![grad; len])
    }
}

impl Scratch {
    pub fn new(stream: &CudaStream, len: usize, gram_dim: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            oriented: DeviceBuffer::<f32>::zeroed(stream, len)?,
            polar_next: DeviceBuffer::<f32>::zeroed(stream, len)?,
            polar_x: DeviceBuffer::<f32>::zeroed(stream, len)?,
            polar_gram: DeviceBuffer::<f32>::zeroed(stream, gram_dim * gram_dim)?,
            polar_ax: DeviceBuffer::<f32>::zeroed(stream, len)?,
            polar_chunks: DeviceBuffer::<f32>::zeroed(stream, 2 * MUON_COOPERATIVE_BLOCKS)?,
        })
    }
}

pub fn descriptors(slots: &Slots, rows: usize, cols: usize) -> Vec<MuonSlotDescriptor> {
    (0..SLOT_COUNT)
        .map(|slot| MuonSlotDescriptor {
            grad: slots.grads[slot].cu_deviceptr(),
            momentum: slots.momentums[slot].cu_deviceptr(),
            variance: 0,
            second_momentum: 0,
            z_master: slots.z_masters[slot].cu_deviceptr(),
            x_master: slots.x_masters[slot].cu_deviceptr(),
            schedule_amax: slots.schedule_amaxes[slot].cu_deviceptr(),
            bytes: slots.bytes[slot].cu_deviceptr(),
            scales: slots.scales[slot].cu_deviceptr(),
            global_scale: slots.global_scales[slot].cu_deviceptr(),
            rows: rows as u32,
            cols: cols as u32,
            learning_rate_multiplier: 1.0,
            qk_clip_factor_offset: u32::MAX,
            qk_clip_head_dim: 0,
        })
        .collect()
}

pub fn assert_quantized_slot_matches(
    stream: &CudaStream,
    mut slots: Slots,
    expected: f32,
) -> Result<(), Box<dyn Error>> {
    let bytes = slots.bytes.remove(0).to_host_vec(stream)?;
    let scales = slots.scales.remove(0).to_host_vec(stream)?;
    let global_scale = slots.global_scales.remove(0).to_host_vec(stream)?;
    crate::common::assert_nvfp4_buffers_nonzero(&bytes, &scales);
    assert!((global_scale[0] - expected.abs() / (256.0 * 6.0)).abs() <= 1.0e-8);
    Ok(())
}

fn slot_buffers(
    stream: &CudaStream,
    values: &[f32],
) -> Result<Vec<DeviceBuffer<f32>>, Box<dyn Error>> {
    (0..SLOT_COUNT)
        .map(|_| DeviceBuffer::from_host(stream, values).map_err(Into::into))
        .collect()
}

fn fill_slot_buffers(
    stream: &CudaStream,
    value: f32,
    len: usize,
) -> Result<Vec<DeviceBuffer<f32>>, Box<dyn Error>> {
    slot_buffers(stream, &vec![value; len])
}

fn zero_slot_buffers<T>(
    stream: &CudaStream,
    len: usize,
) -> Result<Vec<DeviceBuffer<T>>, Box<dyn Error>>
where
    T: DeviceCopy,
{
    (0..SLOT_COUNT)
        .map(|_| DeviceBuffer::<T>::zeroed(stream, len).map_err(Into::into))
        .collect()
}
