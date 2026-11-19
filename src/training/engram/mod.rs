//! Complete single-stream Engram module integrated into the native trainer.
#[cfg(test)]
mod adjoint_verification;
mod backward;
mod backward_multi;
mod backward_scratch;
mod config;
mod forward;
mod modules;
mod state;
mod weights;
pub(crate) use config::Config;
pub(super) use modules::Engram;
pub(crate) use modules::Weights;
pub(super) use state::Module;
pub(crate) use weights::ModuleWeights;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod failure_replay_tests;
#[cfg(test)]
mod forward_oracle_tests;
#[cfg(test)]
mod full_architecture_tests;
#[cfg(test)]
mod materialization_tests;
#[cfg(test)]
pub(in crate::training) mod multi_stream_tests;
#[cfg(test)]
mod reference_recipe_tests;
#[cfg(test)]
mod table_budget_tests;
#[cfg(test)]
mod tests;
use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig, memory};
pub(crate) const HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 64;
pub(crate) const MEMORY: usize = 2 * HEADS * HEAD_DIM;
pub(crate) const LAYER: usize = 1;
pub(crate) const LAYERS: [usize; 2] = [1, 5];
pub(crate) const PRIMES: [u32; 16] = config::reference_primes(0);
pub(crate) const MULTIPLIERS: [u64; 3] = [2102040339213, 101765448293423, 147311892553213];
const fn table_rows() -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < 16 {
        sum += PRIMES[i] as usize;
        i += 1;
    }
    sum
}
pub(crate) const TABLE_ROWS: usize = table_rows();
pub(crate) const TABLE_LEN: usize = (TABLE_ROWS * HEAD_DIM).div_ceil(128) * 128;
pub(crate) fn enabled() -> bool {
    std::env::var("TRAIN_ENGRAM").map_or(true, |s| s != "0")
}
pub(crate) fn multistream_enabled() -> bool {
    std::env::var("TRAIN_ENGRAM_MHC").is_ok_and(|s| s == "1")
}
fn grid(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn rows(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn zero(s: &CudaStream, b: &DeviceBuffer<f32>) -> Result<(), DriverError> {
    unsafe { memory::memset_d8_async(b.cu_deviceptr(), 0, b.num_bytes(), s.cu_stream()) }
}
fn copy(
    s: &CudaStream,
    from: &DeviceBuffer<f32>,
    to: &DeviceBuffer<f32>,
    len: usize,
) -> Result<(), DriverError> {
    unsafe {
        memory::memcpy_dtod_async(
            to.cu_deviceptr(),
            from.cu_deviceptr(),
            len * 4,
            s.cu_stream(),
        )
    }
}
