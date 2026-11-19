use std::sync::Arc;

use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError};

use super::{CROSS_ENTROPY_THREADS_PER_BLOCK, CrossEntropyParams, kernels};
use crate::launch::grid_x_config;

pub struct CrossEntropyArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub logits: &'a DeviceBuffer<f32>,
    pub targets: &'a DeviceBuffer<u32>,
    pub losses: &'out mut DeviceBuffer<f32>,
    pub dlogits: &'out mut DeviceBuffer<f32>,
    pub dlogits_row_amax: &'out mut DeviceBuffer<f32>,
    pub token_count: u32,
    pub vocab_size: u32,
}

pub struct LossModule {
    module: kernels::LoadedModule,
    loss_only: Option<kernels::loss_only::LoadedModule>,
}

impl LossModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        let loss_only = crate::module_exports::contains(&module, b"cross_entropy_loss_kernel")?
            .then(|| kernels::loss_only::from_module(module.clone()))
            .transpose()?;
        Ok(Self {
            module: kernels::from_module(module)?,
            loss_only,
        })
    }

    pub fn try_cross_entropy_loss(
        &self,
        stream: &CudaStream,
        logits: &DeviceBuffer<f32>,
        targets: &DeviceBuffer<u32>,
        losses: &mut DeviceBuffer<f32>,
        params: CrossEntropyParams,
    ) -> Result<bool, DriverError> {
        let Some(module) = &self.loss_only else {
            return Ok(false);
        };
        module.cross_entropy_loss_kernel(
            stream,
            grid_x_config(params.token_count, CROSS_ENTROPY_THREADS_PER_BLOCK),
            logits,
            targets,
            losses,
            params,
        )?;
        Ok(true)
    }

    pub fn cross_entropy(&self, args: CrossEntropyArgs<'_, '_>) -> Result<(), DriverError> {
        self.module.cross_entropy_kernel(
            args.stream,
            grid_x_config(args.token_count, CROSS_ENTROPY_THREADS_PER_BLOCK),
            args.logits,
            args.targets,
            args.losses,
            args.dlogits,
            args.dlogits_row_amax,
            CrossEntropyParams {
                token_count: args.token_count,
                vocab_size: args.vocab_size,
            },
        )
    }
}
