use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{
    OptimizerModule, ScheduleFreeMaterializeArgs, ScheduleFreeMaterializePrecomputedArgs,
    SymExpLinScaleRefs,
};

use crate::upload::UploadedNvfp4;

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{AdamState, MuonState, SymExpLinState, TokenEmbeddingState};

pub(super) struct Materializer<'a> {
    stream: &'a CudaStream,
    optimizer: &'a OptimizerModule,
    scratch: &'a mut OptimizerScratch,
    beta: f32,
    symexp_lin_beta: f32,
}

impl<'a> Materializer<'a> {
    pub(super) fn new(
        stream: &'a CudaStream,
        optimizer: &'a OptimizerModule,
        scratch: &'a mut OptimizerScratch,
        beta: f32,
        symexp_lin_beta: f32,
    ) -> Self {
        Self {
            stream,
            optimizer,
            scratch,
            beta,
            symexp_lin_beta,
        }
    }

    pub(super) fn adam(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &AdamState,
    ) -> Result<(), DriverError> {
        self.tensor(tensor, &state.z_master, &state.x_master, 0.0, None)
    }

    pub(super) fn token_embedding(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &TokenEmbeddingState,
    ) -> Result<(), DriverError> {
        self.tensor(tensor, state.z_master(), state.x_master(), 0.0, None)
    }

    pub(super) fn symexp_lin_adam(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &AdamState,
    ) -> Result<(), DriverError> {
        self.tensor(
            tensor,
            &state.z_master,
            &state.x_master,
            self.symexp_lin_beta,
            None,
        )
    }

    pub(super) fn muon(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &MuonState,
    ) -> Result<(), DriverError> {
        let scales = Self::scale_refs(&state.symexp_lin);
        self.tensor(
            tensor,
            &state.z_master,
            &state.x_master,
            self.symexp_lin_beta,
            scales,
        )
    }

    pub(super) fn muon_precomputed(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &MuonState,
    ) -> Result<(), DriverError> {
        let scales = Self::scale_refs(&state.symexp_lin);
        self.optimizer.materialize_schedule_free_precomputed(
            ScheduleFreeMaterializePrecomputedArgs {
                stream: self.stream,
                bytes: &mut tensor.bytes,
                scales: &mut tensor.scales,
                global_scale: &mut tensor.global_scale,
                z_master: &state.z_master,
                x_master: &state.x_master,
                amax: &state.schedule_amax,
                len: tensor.len as u32,
                beta: self.beta,
                symexp_lin_beta: self.symexp_lin_beta,
                symexp_lin_scales: scales,
            },
        )
    }

    pub(super) fn master(
        &mut self,
        tensor: &mut UploadedNvfp4,
        master: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        self.optimizer.materialize_master(
            self.stream,
            &mut tensor.bytes,
            &mut tensor.scales,
            &mut tensor.global_scale,
            master,
            &mut self.scratch.amax,
            &mut self.scratch.chunk_amax,
            tensor.len as u32,
        )
    }

    pub(super) fn symexp_lin_master(
        &mut self,
        tensor: &mut UploadedNvfp4,
        master: &DeviceBuffer<f32>,
    ) -> Result<(), DriverError> {
        if self.symexp_lin_beta == 0.0 {
            return self.master(tensor, master);
        }
        self.optimizer
            .materialize_schedule_free(ScheduleFreeMaterializeArgs {
                stream: self.stream,
                bytes: &mut tensor.bytes,
                scales: &mut tensor.scales,
                global_scale: &mut tensor.global_scale,
                z_master: master,
                x_master: master,
                amax: &mut self.scratch.amax,
                chunk_amax: &mut self.scratch.chunk_amax,
                len: tensor.len as u32,
                beta: 0.0,
                symexp_lin_beta: self.symexp_lin_beta,
                symexp_lin_scales: None,
            })
    }

    pub(super) fn symexp_lin_muon_master(
        &mut self,
        tensor: &mut UploadedNvfp4,
        master: &DeviceBuffer<f32>,
        state: &SymExpLinState,
    ) -> Result<(), DriverError> {
        if self.symexp_lin_beta == 0.0 {
            return self.master(tensor, master);
        }
        let scales = Self::scale_refs(state);
        self.optimizer
            .materialize_schedule_free(ScheduleFreeMaterializeArgs {
                stream: self.stream,
                bytes: &mut tensor.bytes,
                scales: &mut tensor.scales,
                global_scale: &mut tensor.global_scale,
                z_master: master,
                x_master: master,
                amax: &mut self.scratch.amax,
                chunk_amax: &mut self.scratch.chunk_amax,
                len: tensor.len as u32,
                beta: 1.0,
                symexp_lin_beta: self.symexp_lin_beta,
                symexp_lin_scales: scales,
            })
    }

    fn tensor(
        &mut self,
        tensor: &mut UploadedNvfp4,
        z_master: &DeviceBuffer<f32>,
        x_master: &DeviceBuffer<f32>,
        symexp_lin_beta: f32,
        symexp_lin_scales: Option<SymExpLinScaleRefs<'_>>,
    ) -> Result<(), DriverError> {
        self.optimizer
            .materialize_schedule_free(ScheduleFreeMaterializeArgs {
                stream: self.stream,
                bytes: &mut tensor.bytes,
                scales: &mut tensor.scales,
                global_scale: &mut tensor.global_scale,
                z_master,
                x_master,
                amax: &mut self.scratch.amax,
                chunk_amax: &mut self.scratch.chunk_amax,
                len: tensor.len as u32,
                beta: self.beta,
                symexp_lin_beta,
                symexp_lin_scales,
            })
    }

    fn scale_refs(state: &SymExpLinState) -> Option<SymExpLinScaleRefs<'_>> {
        super::super::symexp_lin::learn_scales().then_some(SymExpLinScaleRefs {
            exponential_z: &state.exponential.z_master,
            exponential_x: &state.exponential.x_master,
            linear_z: &state.linear.z_master,
            linear_x: &state.linear.x_master,
            curvature_z: &state.curvature.z_master,
            curvature_x: &state.curvature.x_master,
        })
    }
}
