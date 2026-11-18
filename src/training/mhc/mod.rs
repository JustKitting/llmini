//! Complete four-stream mHC: 32 branch boundaries and a learned final collapse.
mod activation;
mod backward;
mod forward;
#[cfg(test)]
mod recompute_verification;
mod state;
#[cfg(test)]
mod tests;
mod update;
mod weights;
use cuda_core::{CudaStream, DeviceBuffer, DriverError, LaunchConfig, memory};
use rust_kernels_cuda::mhc_post_reuse::{MhcPostReuseModule, PostArgs};
pub(in crate::training) use state::Mhc;
pub(crate) use weights::{UnitWeights, Weights};
pub(crate) const STREAMS: usize = 4;
pub(crate) const WIDTH: usize = STREAMS * gpt2_nvfp4::GPT2_N_EMBD;
pub(crate) const PAD: usize = 128;
pub(crate) const BOUNDARIES: usize = 2 * gpt2_nvfp4::GPT2_N_LAYER;
pub(crate) const WINDOW: usize = 4;
fn window_start(index: usize) -> usize {
    if index < 2 {
        0
    } else {
        2 + (index - 2) / WINDOW * WINDOW
    }
}
fn window_end(start: usize) -> usize {
    if start == 0 {
        2.min(BOUNDARIES + 1)
    } else {
        (start + WINDOW).min(BOUNDARIES + 1)
    }
}
fn tape_slot(index: usize) -> usize {
    index - window_start(index)
}
pub(crate) fn enabled() -> bool {
    std::env::var("TRAIN_MHC").is_ok_and(|s| s == "1")
}
pub(in crate::training) fn post_reuse_enabled() -> bool {
    super::env::env_bool("TRAIN_MHC_POST_REUSE").unwrap_or(false)
}
pub(in crate::training) fn window_post_enabled() -> bool {
    super::env::env_bool("TRAIN_MHC_WINDOW_POST").unwrap_or(false)
}

fn post_mix(
    reuse: Option<&MhcPostReuseModule>,
    ordinary: &rust_kernels_cuda::mhc::kernels::LoadedModule,
    args: PostArgs<'_, '_>,
) -> Result<(), DriverError> {
    if let Some(reuse) = reuse {
        return reuse.post(args);
    }
    ordinary.mhc_post_kernel(
        args.stream,
        grid(args.rows * args.dim * STREAMS as u32),
        args.input,
        args.branch,
        args.coefficients,
        args.out,
        args.rows,
        args.dim,
    )
}
pub(crate) fn descriptor(version: u32) -> Vec<f32> {
    assert!([1, 2].contains(&version));
    let mut fields = vec![
        version as f32,
        STREAMS as f32,
        gpt2_nvfp4::GPT2_N_EMBD as f32,
        BOUNDARIES as f32,
        20.,
        PAD as f32,
        1e-6,
        0.02,
        if version == 1 { 1. } else { 0.01 },
    ];
    if version == 2 {
        // The paper specifies branch alpha initialization. HyperHead is the
        // pinned library extension and retains its separately documented init.
        fields.push(1.);
    }
    fields
}
fn rows(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn grid(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        ..rows(n)
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
