//! Full detached latent feedback; existing native optimizer and dense NVFP4 paths.
mod state;
mod weights;
pub(in crate::training) use state::Feedback;
pub(crate) use weights::Weights;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod failure_replay_tests;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
pub(crate) const EPSILON: f32 = 1e-5;
pub(crate) const ONE_PASS_PROBABILITY: f32 = 0.15;
pub(crate) const EVAL_PASSES: usize = 4;
fn sample_training_passes(seed: u64, next_attempt: &mut u64) -> usize {
    let attempt = *next_attempt;
    *next_attempt = attempt.wrapping_add(1);
    let mut r =
        gpt2_nvfp4::Gpt2Rng::new(seed ^ attempt.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x4c4f524d);
    if f64::from(r.next_u32()) / 4294967296.0 < f64::from(ONE_PASS_PROBABILITY) {
        1
    } else {
        2
    }
}
pub(super) fn zero(s: &CudaStream, b: &DeviceBuffer<f32>) -> Result<(), DriverError> {
    unsafe { cuda_core::memory::memset_d8_async(b.cu_deviceptr(), 0, b.num_bytes(), s.cu_stream()) }
}
