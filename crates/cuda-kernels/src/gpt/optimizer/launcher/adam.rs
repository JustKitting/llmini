use cuda_core::DriverError;

use super::super::args::AdamWUpdateArgs;
use super::super::threads::APPLY_THREADS_PER_BLOCK;
use super::OptimizerModule;
use crate::launch::linear_config;

impl OptimizerModule {
    pub fn apply_adamw_update(&self, args: AdamWUpdateArgs<'_>) -> Result<(), DriverError> {
        assert_eq!(args.len % 16, 0);
        assert!(args.z_master.len() >= args.len as usize);
        assert!(args.x_master.len() >= args.len as usize);
        assert!(args.grad.len() >= args.len as usize);
        assert!(args.first_moment.len() >= args.len as usize);
        assert!(args.second_moment.len() >= args.len as usize);

        self.apply.adam.fp32_adamw_update_kernel(
            args.stream,
            linear_config(args.len, APPLY_THREADS_PER_BLOCK),
            args.z_master,
            args.x_master,
            args.grad,
            args.grad_scale,
            args.first_moment,
            args.second_moment,
            args.learning_rate,
            args.weight_decay,
            args.beta1,
            args.beta2,
            args.beta1_correction,
            args.beta2_correction,
            args.eps,
            args.average_coefficient,
            args.len,
        )?;

        self.requantize(
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
}
