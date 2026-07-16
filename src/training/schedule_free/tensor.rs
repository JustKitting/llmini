use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{
    OptimizerModule, ScheduleFreeMaterializeArgs, ScheduleFreeMaterializePrecomputedArgs,
};

use crate::upload::UploadedNvfp4;

use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{AdamState, MuonState};

pub(super) struct Materializer<'a> {
    stream: &'a CudaStream,
    optimizer: &'a OptimizerModule,
    scratch: &'a mut OptimizerScratch,
    beta: f32,
}

impl<'a> Materializer<'a> {
    pub(super) fn new(
        stream: &'a CudaStream,
        optimizer: &'a OptimizerModule,
        scratch: &'a mut OptimizerScratch,
        beta: f32,
    ) -> Self {
        Self {
            stream,
            optimizer,
            scratch,
            beta,
        }
    }

    pub(super) fn adam(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &AdamState,
    ) -> Result<(), DriverError> {
        self.tensor(tensor, &state.z_master, &state.x_master)
    }

    pub(super) fn muon(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &MuonState,
    ) -> Result<(), DriverError> {
        self.tensor(tensor, &state.z_master, &state.x_master)
    }

    pub(super) fn muon_precomputed(
        &mut self,
        tensor: &mut UploadedNvfp4,
        state: &MuonState,
    ) -> Result<(), DriverError> {
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

    fn tensor(
        &mut self,
        tensor: &mut UploadedNvfp4,
        z_master: &DeviceBuffer<f32>,
        x_master: &DeviceBuffer<f32>,
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
            })
    }
}
