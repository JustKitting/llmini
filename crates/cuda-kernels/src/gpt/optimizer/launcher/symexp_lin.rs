use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::super::SymExpLinSlotDescriptor;
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::linear_config;

impl OptimizerModule {
    pub fn symexp_lin_zero_scale_grads(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<SymExpLinSlotDescriptor>,
        slot_count: u32,
    ) -> Result<(), DriverError> {
        assert!((slot_count as usize) <= slots.len());
        self.apply.symexp_lin.symexp_lin_zero_scale_grads_kernel(
            stream,
            linear_config(slot_count, APPLY_THREADS_PER_BLOCK),
            slots,
            slot_count,
        )
    }

    #[expect(clippy::too_many_arguments)]
    pub fn symexp_lin_scale_adamw_update(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<SymExpLinSlotDescriptor>,
        slot_count: u32,
        grad_scale: f32,
        exponential_linear_learning_rate: f32,
        curvature_learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
    ) -> Result<(), DriverError> {
        assert!((slot_count as usize) <= slots.len());
        self.apply.symexp_lin.symexp_lin_scale_adamw_update_kernel(
            stream,
            linear_config(slot_count, APPLY_THREADS_PER_BLOCK),
            slots,
            slot_count,
            grad_scale,
            exponential_linear_learning_rate,
            curvature_learning_rate,
            weight_decay,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            average_coefficient,
        )
    }

    pub fn symexp_lin_congruent_inverse_in_place(
        &self,
        stream: &CudaStream,
        raw: &mut DeviceBuffer<f32>,
        beta: f32,
    ) -> Result<(), DriverError> {
        assert!(beta.is_finite() && beta > 0.0);
        let len = raw.len() as u32;
        self.apply.symexp_lin.symexp_lin_congruent_inverse_kernel(
            stream,
            linear_config(len, APPLY_THREADS_PER_BLOCK),
            raw,
            beta,
            len,
        )
    }
}
