mod causal;
mod causal_backward_tc;
mod causal_tc;
mod headwise_gate;
pub(crate) mod layout;
mod qkv_projection;
mod rope;
mod value_residual;
pub(crate) use causal_backward_tc::exact_kda_raw_finish_body;
pub(crate) use causal_tc::{
    exact_kda_output_body, exact_kda_state_body, exact_kda_state_carry_body,
};

use std::sync::{Arc, OnceLock};

use cuda_core::{CudaModule, DriverError};

pub(crate) const QK_NORM_EPS: f32 = 1.0e-12;

pub use causal::{CausalAttentionArgs, CausalAttentionParams};
pub use causal_backward_tc::{CausalAttentionBackwardTcArgs, CausalAttentionBackwardTcScratch};
pub use causal_tc::{CausalAttentionTcArgs, CausalAttentionTcScratch};
pub use headwise_gate::{
    AttentionOutputGate, ExclusiveSelfAttentionBackwardArgs, ExclusiveSelfAttentionForwardArgs,
    HeadwiseAttentionGateBackwardArgs, HeadwiseAttentionGateForwardArgs,
};
pub use qkv_projection::{CProjArgs, QkvProjectionArgs, QkvProjectionParams};
pub use rope::{ApplyRopeArgs, ApplyRopeParams};
pub use value_residual::{
    AccumulateValueResidualGradArgs, CaptureValueResidualArgs, FinishValueResidualGradArgs,
    InitializeValueResidualGradArgs, MixValueResidualArgs,
};

pub struct AttentionModule {
    cuda_module: Arc<CudaModule>,
    full_state_decay:
        OnceLock<Result<causal_backward_tc::full_state_decay::LoadedModule, DriverError>>,
    qkv_projection: qkv_projection::kernels::LoadedModule,
    causal_attention: causal::kernels::LoadedModule,
    causal_attention_backward_tc: causal_backward_tc::kernels::LoadedModule,
    causal_attention_tc: causal_tc::kernels::LoadedModule,
    headwise_gate: headwise_gate::kernels::LoadedModule,
    rope: rope::kernels::LoadedModule,
    value_residual: value_residual::kernels::LoadedModule,
}

impl AttentionModule {
    fn full_state_decay_module(
        &self,
    ) -> Result<&causal_backward_tc::full_state_decay::LoadedModule, DriverError> {
        match self.full_state_decay.get_or_init(|| {
            causal_backward_tc::full_state_decay::from_module(self.cuda_module.clone())
        }) {
            Ok(module) => Ok(module),
            Err(error) => Err(*error),
        }
    }
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            cuda_module: module.clone(),
            full_state_decay: OnceLock::new(),
            qkv_projection: qkv_projection::kernels::from_module(module.clone())?,
            causal_attention: causal::kernels::from_module(module.clone())?,
            causal_attention_backward_tc: causal_backward_tc::kernels::from_module(module.clone())?,
            causal_attention_tc: causal_tc::kernels::from_module(module.clone())?,
            headwise_gate: headwise_gate::kernels::from_module(module.clone())?,
            rope: rope::kernels::from_module(module.clone())?,
            value_residual: value_residual::kernels::from_module(module)?,
        })
    }
}
