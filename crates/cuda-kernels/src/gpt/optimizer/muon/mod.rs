//! Muon device kernels.
//!
//! Algorithm ownership:
//! - `fused`: cooperative Muon matrix update path.
//! - `polar`: shared Polar Express inner loop used by `fused`.

mod fused;
pub(crate) mod polar;

use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

pub(super) struct LoadedModule {
    pub(super) mega: fused::mega::module::LoadedModule,
    pub(super) tma_split: fused::tma_split::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        mega: fused::mega::module::from_module(module.clone())?,
        tma_split: fused::tma_split::module::from_module(module)?,
    })
}
