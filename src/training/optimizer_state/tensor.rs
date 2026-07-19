use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;

use super::device::{clone_device, decode_master};
use crate::{training::device_buffer::zero, upload::UploadedNvfp4};

#[derive(Clone, Copy)]
pub(super) struct StateInit<'a> {
    stream: &'a CudaStream,
    decode: &'a Nvfp4DecodeModule,
}

impl<'a> StateInit<'a> {
    pub(super) fn new(stream: &'a CudaStream, decode: &'a Nvfp4DecodeModule) -> Self {
        Self { stream, decode }
    }
}

pub(in crate::training) struct AdamState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) first: DeviceBuffer<f32>,
    pub(in crate::training) second: DeviceBuffer<f32>,
}

impl AdamState {
    pub(super) fn new(init: StateInit<'_>, tensor: &UploadedNvfp4) -> Result<Self, DriverError> {
        let master = decode_master(init.stream, init.decode, tensor)?;
        Ok(Self {
            z_master: clone_device(init.stream, &master)?,
            x_master: master,
            first: zero(init.stream, tensor.len)?,
            second: zero(init.stream, tensor.len)?,
        })
    }
}

pub(in crate::training) struct MuonState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) momentum: DeviceBuffer<f32>,
    pub(in crate::training) variance: DeviceBuffer<u16>,
    pub(in crate::training) second_momentum: DeviceBuffer<f32>,
    pub(in crate::training) schedule_amax: DeviceBuffer<f32>,
}

impl MuonState {
    pub(super) fn new(
        init: StateInit<'_>,
        tensor: &UploadedNvfp4,
        normuon_neurons: usize,
    ) -> Result<Self, DriverError> {
        let master = decode_master(init.stream, init.decode, tensor)?;
        Ok(Self {
            z_master: clone_device(init.stream, &master)?,
            x_master: master,
            momentum: zero(init.stream, tensor.len)?,
            variance: zero(init.stream, tensor.len)?,
            second_momentum: zero(init.stream, normuon_neurons)?,
            schedule_amax: zero(init.stream, 1)?,
        })
    }
}
