use cuda_core::DeviceBuffer;

use crate::random::InitRng;
use crate::{GPT2_CANON_KERNEL_SIZE, GPT2_N_EMBD};

#[derive(Clone, Debug)]
pub struct CanonWeights {
    pub values: Vec<f32>,
}

impl CanonWeights {
    pub(crate) fn init(rng: &mut InitRng) -> Self {
        let count = GPT2_N_EMBD * GPT2_CANON_KERNEL_SIZE;
        let values = (0..count)
            .map(|_| {
                let unit = rng.next_u32() as f64 / (u32::MAX as f64 + 1.0);
                (unit - 0.5) as f32
            })
            .collect();
        Self { values }
    }
}

#[derive(Clone, Copy)]
pub struct CanonTensors<'a> {
    pub weight: &'a DeviceBuffer<f32>,
}
