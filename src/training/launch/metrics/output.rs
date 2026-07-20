use std::sync::Arc;

use burn::train::metric::Adaptor;

use super::super::super::TrainStats;

macro_rules! impl_output_item {
    ($ty:ty) => {
        impl burn::train::ItemLazy for $ty {
            type ItemSync = Self;

            fn sync(self) -> Self::ItemSync {
                self
            }
        }

        impl Adaptor<$ty> for $ty {
            fn adapt(&self) -> $ty {
                self.clone()
            }
        }
    };
}

#[derive(Clone)]
pub(in crate::training) struct CudaTrainOutput {
    pub(in crate::training) source: String,
    pub(in crate::training) window_offset: usize,
    pub(in crate::training) batch_size: usize,
    pub(in crate::training) seq_len: usize,
    pub(in crate::training) stats: Arc<TrainStats>,
}

impl_output_item!(CudaTrainOutput);

#[derive(Clone)]
pub(in crate::training) struct CudaValidOutput {
    pub(in crate::training::launch) val_loss: f32,
    pub(in crate::training::launch) val_bits_per_byte: f64,
    pub(in crate::training::launch) eval_elapsed_s: f64,
    pub(in crate::training::launch) window_count: usize,
    pub(in crate::training::launch) target_token_count: usize,
    pub(in crate::training::launch) target_byte_count: usize,
    pub(in crate::training::launch) completed_steps: usize,
}

impl_output_item!(CudaValidOutput);
