mod causal;
mod causal_backward_tc;
mod causal_tc;
pub(crate) mod layout;
mod qkv_projection;
mod rope;
mod value_residual;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub(crate) const QK_NORM_EPS: f32 = 1.0e-12;

pub use causal::{CausalAttentionArgs, CausalAttentionParams};
pub use causal_backward_tc::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
pub use causal_tc::{CausalAttentionTcArgs, CausalAttentionTcScratch};
pub use qkv_projection::{CProjArgs, QkvProjectionArgs, QkvProjectionParams};
pub use rope::{ApplyRopeArgs, ApplyRopeParams};
pub use value_residual::{
    AccumulateValueResidualGradArgs, CaptureValueResidualArgs, FinishValueResidualGradArgs,
    InitializeValueResidualGradArgs, MixValueResidualArgs,
};

pub struct AttentionModule {
    qkv_projection: qkv_projection::kernels::LoadedModule,
    causal_attention: causal::kernels::LoadedModule,
    causal_attention_backward_tc: causal_backward_tc::kernels::LoadedModule,
    causal_attention_tc: causal_tc::kernels::LoadedModule,
    rope: rope::kernels::LoadedModule,
    value_residual: value_residual::kernels::LoadedModule,
}

impl AttentionModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            qkv_projection: qkv_projection::kernels::from_module(module.clone())?,
            causal_attention: causal::kernels::from_module(module.clone())?,
            causal_attention_backward_tc: causal_backward_tc::kernels::from_module(module.clone())?,
            causal_attention_tc: causal_tc::kernels::from_module(module.clone())?,
            rope: rope::kernels::from_module(module.clone())?,
            value_residual: value_residual::kernels::from_module(module)?,
        })
    }
}
