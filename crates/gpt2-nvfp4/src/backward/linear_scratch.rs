use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::linear_backward::{
    LinearBackwardMsEdenScratch, LinearBackwardMsEdenScratchBuffers,
};

use super::attention::{AttentionCProjScratch, AttentionQkvScratch};
use super::final_head::FinalHeadBackwardScratch;
use crate::GPT2_TOKEN_ROWS;

pub struct LinearScratch {
    linear: LinearBackwardMsEdenScratchBuffers,
}

impl LinearScratch {
    pub fn new(
        stream: &CudaStream,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            linear: LinearBackwardMsEdenScratchBuffers::new(
                stream,
                GPT2_TOKEN_ROWS,
                input_dim,
                output_dim,
            )?,
        })
    }

    pub fn c_proj(&mut self) -> AttentionCProjScratch<'_> {
        AttentionCProjScratch {
            linear: self.parts(),
        }
    }

    pub fn qkv(&mut self) -> AttentionQkvScratch<'_> {
        self.c_proj()
    }

    pub fn final_head(&mut self) -> FinalHeadBackwardScratch<'_> {
        FinalHeadBackwardScratch {
            linear: self.parts(),
        }
    }

    pub fn parts(&mut self) -> LinearBackwardMsEdenScratch<'_> {
        self.linear.as_args()
    }
}
