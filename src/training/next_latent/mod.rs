mod backward;
mod backward_linear;
mod backward_linear_call;
mod backward_norm;
mod buffers;
mod forward;
mod grads;
mod projection;
mod quantize;
mod scratch;

pub use backward::{NextLatBackwardArgs, NextLatBackwardSeeds, backward};
pub use buffers::NextLatBuffers;
pub use forward::{NextLatForwardArgs, forward};
pub use grads::NextLatGradBuffers;
pub use scratch::NextLatScratchBuffers;

use std::sync::OnceLock;

use super::env::env_f32;

pub(super) fn loss_weight() -> f32 {
    static LOSS_WEIGHT: OnceLock<f32> = OnceLock::new();
    *LOSS_WEIGHT.get_or_init(|| {
        env_f32("TRAIN_NEXTLAT_LOSS_WEIGHT")
            .filter(|value| value.is_finite())
            .unwrap_or(0.5)
            .clamp(0.1, 4.0)
    })
}
