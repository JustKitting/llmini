use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::optimizer::OptimizerModule;

use super::device::{clone_device, decode_master};
use crate::{training::device_buffer::zero, upload::UploadedNvfp4};

#[derive(Clone, Copy)]
pub(super) struct StateInit<'a> {
    stream: &'a CudaStream,
    decode: &'a Nvfp4DecodeModule,
    optimizer: &'a OptimizerModule,
    symexp_lin_beta: f32,
}

impl<'a> StateInit<'a> {
    pub(super) fn new(
        stream: &'a CudaStream,
        decode: &'a Nvfp4DecodeModule,
        optimizer: &'a OptimizerModule,
        symexp_lin_beta: f32,
    ) -> Self {
        Self {
            stream,
            decode,
            optimizer,
            symexp_lin_beta,
        }
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
        Self::new_inner(init, tensor, false)
    }

    pub(super) fn new_symexp_lin(
        init: StateInit<'_>,
        tensor: &UploadedNvfp4,
    ) -> Result<Self, DriverError> {
        Self::new_inner(init, tensor, true)
    }

    fn new_inner(
        init: StateInit<'_>,
        tensor: &UploadedNvfp4,
        reparameterize: bool,
    ) -> Result<Self, DriverError> {
        let mut master = decode_master(init.stream, init.decode, tensor)?;
        if reparameterize && init.symexp_lin_beta > 0.0 {
            init.optimizer.symexp_lin_congruent_inverse_in_place(
                init.stream,
                &mut master,
                init.symexp_lin_beta,
            )?;
        }
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
    pub(in crate::training) symexp_lin: SymExpLinState,
}

impl MuonState {
    pub(super) fn new(
        init: StateInit<'_>,
        tensor: &UploadedNvfp4,
        normuon_neurons: usize,
    ) -> Result<Self, DriverError> {
        let mut master = decode_master(init.stream, init.decode, tensor)?;
        if init.symexp_lin_beta > 0.0 {
            init.optimizer.symexp_lin_congruent_inverse_in_place(
                init.stream,
                &mut master,
                init.symexp_lin_beta,
            )?;
        }
        Ok(Self {
            z_master: clone_device(init.stream, &master)?,
            x_master: master,
            momentum: zero(init.stream, tensor.len)?,
            variance: zero(init.stream, tensor.len)?,
            second_momentum: zero(init.stream, normuon_neurons)?,
            schedule_amax: zero(init.stream, 1)?,
            symexp_lin: SymExpLinState::new(init.stream)?,
        })
    }
}

pub(in crate::training) struct SymExpLinState {
    pub(in crate::training) exponential: SymExpLinScalarState,
    pub(in crate::training) linear: SymExpLinScalarState,
    pub(in crate::training) curvature: SymExpLinScalarState,
}

impl SymExpLinState {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            exponential: SymExpLinScalarState::new(stream)?,
            linear: SymExpLinScalarState::new(stream)?,
            curvature: SymExpLinScalarState::new(stream)?,
        })
    }
}

pub(in crate::training) struct SymExpLinScalarState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) first: DeviceBuffer<f32>,
    pub(in crate::training) second: DeviceBuffer<f32>,
    pub(in crate::training) grad: DeviceBuffer<f32>,
}

impl SymExpLinScalarState {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            z_master: DeviceBuffer::from_host(stream, &[1.0])?,
            x_master: DeviceBuffer::from_host(stream, &[1.0])?,
            first: zero(stream, 1)?,
            second: zero(stream, 1)?,
            grad: zero(stream, 1)?,
        })
    }
}
