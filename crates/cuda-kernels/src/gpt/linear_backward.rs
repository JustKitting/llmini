use std::sync::Arc;
use std::{cell::RefCell, collections::HashMap};

use cuda_core::{CudaModule, DriverError};

use crate::nvfp4_tma_matmul::gradient::GemmGradientTransform;
use crate::nvfp4_tma_matmul::launcher::Nvfp4GemmModule;

#[path = "linear_backward/args.rs"]
mod args;
#[path = "linear_backward/bias.rs"]
mod bias;
#[path = "linear_backward/device_scale.rs"]
mod device_scale;
#[path = "linear_backward/kernels.rs"]
mod kernels;
#[path = "linear_backward/ms_eden.rs"]
mod ms_eden;
pub use args::{
    LinearBackwardArgs, LinearBackwardDeviceScaleArgs, LinearBackwardInputTranspose,
    LinearBackwardMsEdenArgs, LinearBackwardMsEdenScratch, LinearBackwardMsEdenScratchBuffers,
    LinearBackwardMsEdenWeightArgs, LinearBackwardRoute, LinearBackwardTmaScratch,
    LinearBackwardWeightTranspose, MsEdenOperandScratch, MsEdenOperandScratchBuffer,
    MsEdenTransposeScratch, MsEdenTransposeScratchBuffer,
};
pub use bias::LINEAR_BIAS_THREADS_PER_BLOCK;

pub struct LinearBackwardModule {
    module: kernels::LoadedModule,
    tma: Nvfp4GemmModule,
    weight_gradient_transforms: RefCell<HashMap<u64, GemmGradientTransform>>,
}

impl LinearBackwardModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: kernels::from_module(module.clone())?,
            tma: Nvfp4GemmModule::from_module(module.clone())?,
            weight_gradient_transforms: RefCell::new(HashMap::new()),
        })
    }

    /// Configure the current backward pass using existing optimizer pointers.
    /// The owner keeps these buffers alive through the enqueued backward work.
    pub fn set_weight_gradient_transforms(
        &self,
        transforms: impl IntoIterator<Item = (u64, GemmGradientTransform)>,
    ) {
        let mut targets = self.weight_gradient_transforms.borrow_mut();
        targets.clear();
        for (ptr, transform) in transforms {
            assert!(targets.insert(ptr, transform).is_none());
        }
    }

    pub fn backward(&self, args: LinearBackwardArgs<'_, '_>) -> Result<(), DriverError> {
        let LinearBackwardArgs {
            stream,
            e_h,
            weight_t_h,
            e_t_h,
            input_t_h,
            dinput,
            dweight,
            dbias: _,
            token_count,
            input_dim,
            output_dim,
        } = args;

        self.backward_device_scale(LinearBackwardDeviceScaleArgs {
            stream,
            e_h,
            weight_t_h: weight_t_h.into(),
            e_t_h,
            input_t_h: input_t_h.into(),
            dinput,
            dweight,
            token_count,
            input_dim,
            output_dim,
        })
    }
}
