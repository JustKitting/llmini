use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

mod base;
mod kda_elementwise;
mod kda_tc;

pub(super) use kda_elementwise::KDA_NORM_REDUCE_THREADS_PER_BLOCK;

pub(crate) struct LoadedModule {
    pub(super) base: base::LoadedModule,
    pub(super) kda_elementwise: kda_elementwise::LoadedModule,
    pub(super) kda_tc: kda_tc::LoadedModule,
}

pub(crate) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        base: base::from_module(module.clone())?,
        kda_elementwise: kda_elementwise::from_module(module.clone())?,
        kda_tc: kda_tc::from_module(module)?,
    })
}
