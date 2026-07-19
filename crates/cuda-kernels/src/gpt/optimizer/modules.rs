use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::{adam, embedding, ember, grad_clip, kda_clip, muon, schedule_free, symexp_lin};

pub(super) struct LoadedModule {
    pub(super) adam: adam::module::LoadedModule,
    pub(super) ember: ember::module::LoadedModule,
    pub(super) muon: muon::LoadedModule,
    pub(super) embedding: embedding::module::LoadedModule,
    pub(super) grad_clip: grad_clip::module::LoadedModule,
    pub(super) kda_clip: kda_clip::module::LoadedModule,
    pub(super) schedule_free: schedule_free::module::LoadedModule,
    pub(super) symexp_lin: symexp_lin::module::LoadedModule,
}

pub(super) fn from_module(module: Arc<CudaModule>) -> Result<LoadedModule, DriverError> {
    Ok(LoadedModule {
        adam: adam::module::from_module(module.clone())?,
        ember: ember::module::from_module(module.clone())?,
        muon: muon::from_module(module.clone())?,
        embedding: embedding::module::from_module(module.clone())?,
        grad_clip: grad_clip::module::from_module(module.clone())?,
        kda_clip: kda_clip::module::from_module(module.clone())?,
        schedule_free: schedule_free::module::from_module(module.clone())?,
        symexp_lin: symexp_lin::module::from_module(module)?,
    })
}
