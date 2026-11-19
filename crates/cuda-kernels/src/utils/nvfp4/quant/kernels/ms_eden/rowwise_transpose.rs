use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

#[path = "rowwise_transpose/padded.rs"]
pub(crate) mod padded;

pub(crate) struct LoadedModule {
    pub(crate) padded: padded::LoadedModule,
}

pub(crate) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        padded: padded::from_module(module)?,
    })
}
