use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::training::grad_block::LayerNormGradBuffers;
use crate::training::optimizer_apply::adam_debug_config;
use crate::training::optimizer_state::{AdamState, LayerNormState};
use crate::upload::{UploadedLayerNorm, UploadedNvfp4};

use super::snapshot::{AdamSnapshot, PendingTensorUpdateDiagnostics};

pub(super) struct UpdateSnapshotCollector<'a> {
    stream: &'a CudaStream,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
    updates: Vec<PendingTensorUpdateDiagnostics>,
}

impl<'a> UpdateSnapshotCollector<'a> {
    pub(super) fn new(
        stream: &'a CudaStream,
        step: u32,
        average_coefficient: f32,
        grad_scale: f32,
    ) -> Self {
        Self {
            stream,
            step,
            average_coefficient,
            grad_scale,
            updates: Vec::new(),
        }
    }

    pub(super) fn into_updates(self) -> Vec<PendingTensorUpdateDiagnostics> {
        self.updates
    }

    pub(super) fn push_layer_norm(
        &mut self,
        name: &str,
        layer_norm: &UploadedLayerNorm,
        grads: &LayerNormGradBuffers,
        state: &LayerNormState,
    ) -> AppResult {
        self.push_adam(
            &format!("{name}.weight"),
            &layer_norm.weight,
            &grads.d_weight,
            &state.weight,
        )?;
        self.push_adam(
            &format!("{name}.bias"),
            &layer_norm.bias,
            &grads.d_bias,
            &state.bias,
        )
    }

    pub(super) fn push_adam(
        &mut self,
        name: &str,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &AdamState,
    ) -> AppResult {
        let weight_decay = adam_debug_config(self.step).weight_decay;
        self.push_adam_with_weight_decay(name, tensor, grad, state, weight_decay)
    }

    pub(super) fn push_adam_with_weight_decay(
        &mut self,
        name: &str,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        state: &AdamState,
        weight_decay: f32,
    ) -> AppResult {
        let config = adam_debug_config(self.step);
        let adam = AdamSnapshot {
            z_master: state.z_master.to_host_vec(self.stream)?,
            x_master: state.x_master.to_host_vec(self.stream)?,
            first: state.first.to_host_vec(self.stream)?,
            second: state.second.to_host_vec(self.stream)?,
            learning_rate: config.learning_rate,
            weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            beta1_correction: config.beta1_correction,
            beta2_correction: config.beta2_correction,
            eps: config.eps,
            average_coefficient: self.average_coefficient,
        };
        self.push_snapshot(name, tensor, grad, Some(adam))
    }

    pub(super) fn push_observed(
        &mut self,
        name: &str,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
    ) -> AppResult {
        self.push_snapshot(name, tensor, grad, None)
    }

    fn push_snapshot(
        &mut self,
        name: &str,
        tensor: &UploadedNvfp4,
        grad: &DeviceBuffer<f32>,
        adam: Option<AdamSnapshot>,
    ) -> AppResult {
        let mut grad = grad.to_host_vec(self.stream)?;
        for value in &mut grad {
            *value *= self.grad_scale;
        }
        self.updates.push(PendingTensorUpdateDiagnostics {
            name: name.to_string(),
            len: tensor.len,
            before_bytes: tensor.bytes.to_host_vec(self.stream)?,
            before_scales: tensor.scales.to_host_vec(self.stream)?,
            before_global: tensor.global_scale_to_host(self.stream)?,
            grad,
            adam,
        });
        Ok(())
    }
}
