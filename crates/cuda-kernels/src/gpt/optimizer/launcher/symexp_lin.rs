use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::super::{MuonSlotDescriptor, SymExpLinSlotDescriptor};
use super::OptimizerModule;
use crate::launch::{launch_config, linear_config};

const SYMEXP_LIN_REDUCTION_BLOCKS: u32 = 500;

impl OptimizerModule {
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

    pub fn symexp_lin_chain_rule(
        &self,
        stream: &CudaStream,
        grad: &mut DeviceBuffer<f32>,
        z_master: &DeviceBuffer<f32>,
        x_master: &DeviceBuffer<f32>,
        schedule_beta: f32,
        beta: f32,
    ) -> Result<(), DriverError> {
        assert_eq!(grad.len(), z_master.len());
        assert_eq!(grad.len(), x_master.len());
        assert!(beta.is_finite() && beta > 0.0);
        let len = grad.len() as u32;
        self.apply.symexp_lin.symexp_lin_chain_rule_kernel(
            stream,
            linear_config(len, APPLY_THREADS_PER_BLOCK),
            grad,
            z_master,
            x_master,
            schedule_beta,
            beta,
            len,
        )
    }

    pub fn symexp_lin_slot_chain_rule(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<MuonSlotDescriptor>,
        slot_index: u32,
        matrix_len: u32,
        schedule_beta: f32,
        beta: f32,
    ) -> Result<(), DriverError> {
        assert!((slot_index as usize) < slots.len());
        assert!(matrix_len > 0);
        assert!(beta.is_finite() && beta > 0.0);
        self.apply.symexp_lin.symexp_lin_slot_chain_rule_kernel(
            stream,
            linear_config(matrix_len, APPLY_THREADS_PER_BLOCK),
            slots,
            slot_index,
            schedule_beta,
            beta,
        )
    }

    pub fn symexp_lin_scaled_slot_chain_rule(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<SymExpLinSlotDescriptor>,
        slot_index: u32,
        matrix_len: u32,
        schedule_beta: f32,
        beta: f32,
    ) -> Result<(), DriverError> {
        assert!((slot_index as usize) < slots.len());
        assert!(matrix_len > 0);
        assert!(beta.is_finite() && beta > 0.0);
        let blocks = matrix_len
            .div_ceil(APPLY_THREADS_PER_BLOCK)
            .min(SYMEXP_LIN_REDUCTION_BLOCKS);
        self.apply
            .symexp_lin
            .symexp_lin_scaled_slot_chain_rule_kernel(
                stream,
                launch_config((blocks, 1, 1), APPLY_THREADS_PER_BLOCK),
                slots,
                slot_index,
                schedule_beta,
                beta,
            )
    }
}
