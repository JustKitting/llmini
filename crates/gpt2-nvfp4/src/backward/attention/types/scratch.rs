use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::CausalAttentionBackwardTcScratch;
use rust_kernels_cuda::linear_backward::LinearBackwardMsEdenScratch;

use crate::backward::scratch_reborrow::reborrow_ms_eden;

pub struct AttentionLinearScratch<'scratch> {
    pub linear: LinearBackwardMsEdenScratch<'scratch>,
}

pub type AttentionCProjScratch<'scratch> = AttentionLinearScratch<'scratch>;
pub type AttentionQkvScratch<'scratch> = AttentionLinearScratch<'scratch>;

pub struct AttentionCoreScratch<'scratch> {
    pub softmax_d: &'scratch mut DeviceBuffer<f32>,
    pub tc: CausalAttentionBackwardTcScratch<'scratch>,
}

impl<'scratch> AttentionLinearScratch<'scratch> {
    pub fn reborrow(&mut self) -> AttentionLinearScratch<'_> {
        AttentionLinearScratch {
            linear: reborrow_ms_eden(&mut self.linear),
        }
    }
}

impl<'scratch> AttentionCoreScratch<'scratch> {
    pub fn reborrow(&mut self) -> AttentionCoreScratch<'_> {
        AttentionCoreScratch {
            softmax_d: &mut *self.softmax_d,
            tc: self.tc.reborrow(),
        }
    }
}
