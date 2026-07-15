//! Device optimizer kernels and launch wrappers.
//!
//! Folder ownership:
//! - `adam`: AdamW updates for scalar/vector weights where Muon does not apply.
//! - `muon`: Muon updates for matrix-shaped weights.
//! - `embedding`: token-embedding gradient scatter from residual gradients.
//! - `grad_clip`: global-norm clipping over parameter-gradient buffers.
//! - `schedule_free`: z/x interpolation and materialization for schedule-free state.
//! - `launcher`: host-side CUDA launch wrappers around the device kernels.
//! - `modules`: CUDA module loading registry.
//! - `threads`: shared launch-size constants.

mod adam;
mod args;
mod embedding;
mod grad_clip;
mod kda_clip;
mod launcher;
mod modules;
mod muon;
mod schedule_free;
mod threads;
mod work_grid;

pub use args::{
    AdamWUpdateArgs, EmbeddingLookupGradArgs, GradientClipArgs, KdaMuonClipArgs,
    MuonMegaUpdateArgs, MuonSlotDescriptor, MuonTmaFinishArgs, MuonTmaPrepareArgs,
    ScheduleFreeMaterializeArgs,
};
pub use grad_clip::GRAD_CLIP_VALUES_PER_CHUNK;
pub use launcher::OptimizerModule;
pub use muon::polar::fused::{
    Coefficients as MuonPolarCoefficients, coefficients as muon_polar_coefficients,
};

include!(concat!(env!("OUT_DIR"), "/optimizer_config.rs"));
