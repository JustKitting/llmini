use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_VOCAB_SIZE};
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::optimizer::{OptimizerModule, ember_column_partial_len};

use super::device::{clone_device, decode_master};
use crate::{
    training::device_buffer::zero,
    upload::{UploadedCanon, UploadedNvfp4},
};

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

pub(in crate::training) struct Fp32AdamState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) first: DeviceBuffer<f32>,
    pub(in crate::training) second: DeviceBuffer<f32>,
}

impl Fp32AdamState {
    pub(super) fn new(init: StateInit<'_>, tensor: &UploadedCanon) -> Result<Self, DriverError> {
        Ok(Self {
            z_master: clone_device(init.stream, &tensor.weight)?,
            x_master: clone_device(init.stream, &tensor.weight)?,
            first: zero(init.stream, tensor.weight.len())?,
            second: zero(init.stream, tensor.weight.len())?,
        })
    }
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

pub(in crate::training) enum TokenEmbeddingState {
    Adam(AdamState),
    Ember(EmberState),
}

impl TokenEmbeddingState {
    pub(super) fn new(init: StateInit<'_>, tensor: &UploadedNvfp4) -> Result<Self, DriverError> {
        if super::super::optimizer_apply::ember::enabled() {
            Ok(Self::Ember(EmberState::new(init, tensor)?))
        } else {
            Ok(Self::Adam(AdamState::new(init, tensor)?))
        }
    }

    pub(in crate::training) fn z_master(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Adam(state) => &state.z_master,
            Self::Ember(state) => &state.z_master,
        }
    }

    pub(in crate::training) fn x_master(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Adam(state) => &state.x_master,
            Self::Ember(state) => &state.x_master,
        }
    }

    pub(in crate::training) fn as_adam(&self) -> Option<&AdamState> {
        match self {
            Self::Adam(state) => Some(state),
            Self::Ember(_) => None,
        }
    }
}

pub(in crate::training) struct EmberState {
    pub(in crate::training) z_master: DeviceBuffer<f32>,
    pub(in crate::training) x_master: DeviceBuffer<f32>,
    pub(in crate::training) row_second_moment: DeviceBuffer<f32>,
    pub(in crate::training) column_second_moment: DeviceBuffer<f32>,
    pub(in crate::training) column_partials: DeviceBuffer<f32>,
    pub(in crate::training) normalizer: DeviceBuffer<f32>,
}

impl EmberState {
    fn new(init: StateInit<'_>, tensor: &UploadedNvfp4) -> Result<Self, DriverError> {
        let master = decode_master(init.stream, init.decode, tensor)?;
        let rows = GPT2_VOCAB_SIZE as u32;
        let cols = GPT2_N_EMBD as u32;
        assert_eq!(tensor.len, rows as usize * cols as usize);
        Ok(Self {
            z_master: clone_device(init.stream, &master)?,
            x_master: master,
            row_second_moment: zero(init.stream, rows as usize)?,
            column_second_moment: zero(init.stream, cols as usize)?,
            column_partials: zero(init.stream, ember_column_partial_len(rows, cols))?,
            normalizer: zero(init.stream, 1)?,
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
