use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::MuonSlotDescriptor;

mod host_ptrs;
mod padding;
mod ptrs;
mod slots;
mod table;
use host_ptrs::HostPtrs;
use padding::MuonPaddingBuffers;
use slots::build_slots;
use table::upload_table;

pub(in crate::training) struct MuonPointerTables {
    pub(in crate::training) all: MuonGroupTable,
    pub(in crate::training) slot_count: usize,
    _padding: MuonPaddingBuffers,
}

pub(in crate::training) struct MuonGroupTable {
    pub(super) slots: DeviceBuffer<MuonSlotDescriptor>,
    pub(super) host_slots: Vec<MuonSlotDescriptor>,
}

impl MuonPointerTables {
    pub(in crate::training) fn new(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
    ) -> Result<Self, DriverError> {
        let padding = MuonPaddingBuffers::new(stream)?;
        let rows = build_slots(uploaded, grads, next_latent_grads, state, &padding);
        Ok(Self {
            all: upload_table(stream, &rows)?,
            slot_count: rows.len(),
            _padding: padding,
        })
    }
}
