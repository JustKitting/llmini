use cuda_core::DriverError;

use super::super::args::{AdamWUpdateArgs, Fp32AdamWUpdateArgs};
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::linear_config;

impl OptimizerModule {
    pub fn apply_adamw_update(&self, mut args: AdamWUpdateArgs<'_>) -> Result<(), DriverError> {
        self.apply_adamw_update_master(&mut args)?;
        self.materialize_master(
            args.stream,
            args.bytes,
            args.scales,
            args.global_scale,
            &*args.x_master,
            args.amax,
            args.chunk_amax,
            args.len,
        )
    }

    pub fn apply_adamw_update_deferred_quantization(
        &self,
        mut args: AdamWUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        self.apply_adamw_update_master(&mut args)
    }

    pub fn apply_fp32_adamw_update(
        &self,
        mut args: Fp32AdamWUpdateArgs<'_>,
    ) -> Result<(), DriverError> {
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.grad.len() >= args.len as usize);
        assert!(args.first_moment.len() >= args.len as usize);
        assert!(args.second_moment.len() >= args.len as usize);
        self.launch_fp32_adamw_update(
            args.stream,
            &mut args.z_master,
            &mut args.x_master,
            args.grad,
            args.grad_scale,
            &mut args.first_moment,
            &mut args.second_moment,
            args.learning_rate,
            args.weight_decay,
            args.beta1,
            args.beta2,
            args.beta1_correction,
            args.beta2_correction,
            args.eps,
            args.average_coefficient,
            args.len,
        )
    }

    fn apply_adamw_update_master(&self, args: &mut AdamWUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.grad.len() >= args.len as usize);
        assert!(args.first_moment.len() >= args.len as usize);
        assert!(args.second_moment.len() >= args.len as usize);

        self.launch_fp32_adamw_update(
            args.stream,
            &mut *args.z_master,
            &mut *args.x_master,
            args.grad,
            args.grad_scale,
            &mut *args.first_moment,
            &mut *args.second_moment,
            args.learning_rate,
            args.weight_decay,
            args.beta1,
            args.beta2,
            args.beta1_correction,
            args.beta2_correction,
            args.eps,
            args.average_coefficient,
            args.len,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "CUDA ABI uses explicit optimizer state"
    )]
    fn launch_fp32_adamw_update(
        &self,
        stream: &cuda_core::CudaStream,
        z_master: &mut cuda_core::DeviceBuffer<f32>,
        x_master: &mut cuda_core::DeviceBuffer<f32>,
        grad: &cuda_core::DeviceBuffer<f32>,
        grad_scale: f32,
        first_moment: &mut cuda_core::DeviceBuffer<f32>,
        second_moment: &mut cuda_core::DeviceBuffer<f32>,
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
        len: u32,
    ) -> Result<(), DriverError> {
        self.apply.adam.fp32_adamw_update_kernel(
            stream,
            linear_config(len, APPLY_THREADS_PER_BLOCK),
            z_master,
            x_master,
            grad,
            grad_scale,
            first_moment,
            second_moment,
            learning_rate,
            weight_decay,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            average_coefficient,
            len,
        )
    }
}
