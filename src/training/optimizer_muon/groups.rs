use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer_state::OptimizerStateBuffers;
use crate::upload::UploadedModel;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::optimizer::{MuonSlotDescriptor, SymExpLinSlotDescriptor};

mod host_ptrs;
mod padding;
mod ptrs;
mod slots;
#[cfg(test)]
mod storage_tests;
mod table;
use host_ptrs::HostPtrs;
use padding::MuonPaddingBuffers;
use slots::build_slots;
use table::upload_table;

pub(in crate::training) struct MuonPointerTables {
    pub(in crate::training) all: MuonGroupTable,
    pub(in crate::training) slot_count: usize,
    _padding: MuonPaddingBuffers,
    conditional: Option<ConditionalDescriptors>,
    pub(in crate::training) backbone_only: bool,
}

struct ConditionalDescriptors {
    grad: u64,
    index: usize,
    original: MuonSlotDescriptor,
    enabled: bool,
    slots: DeviceBuffer<MuonSlotDescriptor>,
    symexp: DeviceBuffer<SymExpLinSlotDescriptor>,
}

pub(in crate::training) struct MuonGroupTable {
    pub(super) slots: DeviceBuffer<MuonSlotDescriptor>,
    pub(super) host_slots: Vec<MuonSlotDescriptor>,
    pub(super) symexp_lin_slots: DeviceBuffer<SymExpLinSlotDescriptor>,
    pub(in crate::training) host_symexp_lin_slots: Vec<SymExpLinSlotDescriptor>,
}

impl MuonPointerTables {
    pub(in crate::training) fn new(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
    ) -> Result<Self, DriverError> {
        Self::new_with_engram(stream, uploaded, grads, next_latent_grads, state, None)
    }

    pub(in crate::training) fn new_with_engram(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
        engram: Option<&crate::training::engram::Engram>,
    ) -> Result<Self, DriverError> {
        Self::new_with_modules(
            stream,
            uploaded,
            grads,
            next_latent_grads,
            state,
            engram,
            None,
        )
    }

    /// A residual backbone has no token or NextLat objective. Their matrices
    /// must be absent from both the Muon update and its weight-decay pass.
    pub(in crate::training) fn new_backbone(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        state: &OptimizerStateBuffers,
        mhc: Option<&crate::training::mhc::Mhc>,
    ) -> Result<Self, DriverError> {
        let padding = MuonPaddingBuffers::new(stream)?;
        let rows = build_slots(uploaded, grads, None, state, &padding, None, None, mhc);
        Ok(Self {
            all: upload_table(stream, &rows)?,
            slot_count: rows.len(),
            _padding: padding,
            conditional: None,
            backbone_only: true,
        })
    }

    pub(in crate::training) fn new_with_modules(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
        engram: Option<&crate::training::engram::Engram>,
        feedback: Option<&crate::training::latent_feedback::Feedback>,
    ) -> Result<Self, DriverError> {
        Self::new_with_architectures(
            stream,
            uploaded,
            grads,
            next_latent_grads,
            state,
            engram,
            feedback,
            None,
        )
    }
    pub(in crate::training) fn new_with_architectures(
        stream: &CudaStream,
        uploaded: &UploadedModel,
        grads: &BackwardBuffers,
        next_latent_grads: &NextLatGradBuffers,
        state: &OptimizerStateBuffers,
        engram: Option<&crate::training::engram::Engram>,
        feedback: Option<&crate::training::latent_feedback::Feedback>,
        mhc: Option<&crate::training::mhc::Mhc>,
    ) -> Result<Self, DriverError> {
        let padding = MuonPaddingBuffers::new(stream)?;
        let rows = build_slots(
            uploaded,
            grads,
            Some(next_latent_grads),
            state,
            &padding,
            engram,
            feedback,
            mhc,
        );
        let mut result = Self {
            all: upload_table(stream, &rows)?,
            slot_count: rows.len(),
            _padding: padding,
            conditional: None,
            backbone_only: false,
        };
        if let Some(f) = feedback {
            let i = result
                .all
                .host_slots
                .iter()
                .position(|v| v.grad == f.grad.cu_deviceptr())
                .expect("feedback matrix must be registered");
            let mut slots = result.all.host_slots.clone();
            let mut symexp = result.all.host_symexp_lin_slots.clone();
            slots[i].rows = 0;
            slots[i].cols = 0;
            symexp[i].rows = 0;
            symexp[i].cols = 0;
            symexp[i].exponential_z = 0;
            let disabled = ConditionalDescriptors {
                grad: f.grad.cu_deviceptr(),
                index: i,
                original: result.all.host_slots[i],
                enabled: true,
                slots: DeviceBuffer::from_host(stream, &slots)?,
                symexp: DeviceBuffer::from_host(stream, &symexp)?,
            };
            stream.synchronize()?; // Temporary upload sources remain alive through completion.
            result.conditional = Some(disabled);
        }
        Ok(result)
    }

    pub(in crate::training) fn set_conditional_matrix_enabled(
        &mut self,
        grad: u64,
        enabled: bool,
    ) -> crate::AppResult {
        let c = self
            .conditional
            .as_mut()
            .ok_or("conditional feedback descriptors missing")?;
        if c.grad != grad {
            return Err("conditional feedback pointer does not match registered matrix".into());
        }
        if c.enabled != enabled {
            // Both immutable device tables stay allocated. Queued kernels retain valid pointers.
            std::mem::swap(&mut self.all.slots, &mut c.slots);
            std::mem::swap(&mut self.all.symexp_lin_slots, &mut c.symexp);
            self.all.host_slots[c.index] = c.original;
            if !enabled {
                self.all.host_slots[c.index].rows = 0;
                self.all.host_slots[c.index].cols = 0;
            }
            c.enabled = enabled;
        }
        Ok(())
    }

    pub(in crate::training) fn matrix_grad_ranges(&self) -> Vec<(u64, u32)> {
        self.all
            .host_slots
            .iter()
            .filter_map(|desc| {
                let len = desc.rows * desc.cols;
                (len > 0).then_some((desc.grad, len))
            })
            .collect()
    }

    pub(in crate::training) fn symexp_lin_slots(&self) -> &DeviceBuffer<SymExpLinSlotDescriptor> {
        &self.all.symexp_lin_slots
    }
}
