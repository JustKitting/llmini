use std::sync::Arc;

use cuda_core::{CudaModule, DriverError};

use super::kernels;

pub struct Nvfp4QuantModule {
    pub(super) ms_eden_csf: kernels::ms_eden::csf::module::LoadedModule,
    pub(super) row_amax: kernels::row_amax::module::LoadedModule,
    pub(super) four_six: kernels::four_six::module::LoadedModule,
    pub(super) ms_eden_amax: kernels::ms_eden::amax::module::LoadedModule,
    pub(super) ms_eden_fp32: kernels::ms_eden::fp32::module::LoadedModule,
    pub(super) ms_eden_fp32_transpose: kernels::ms_eden::fp32_transpose::module::LoadedModule,
    pub(super) ms_eden_nvfp4_transpose: kernels::ms_eden::nvfp4_transpose::module::LoadedModule,
    pub(super) ms_eden_rowwise_transpose: kernels::ms_eden::rowwise_transpose::LoadedModule,
}

impl Nvfp4QuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            ms_eden_csf: kernels::ms_eden::csf::module::from_module(module.clone())?,
            row_amax: kernels::row_amax::module::from_module(module.clone())?,
            four_six: kernels::four_six::module::from_module(module.clone())?,
            ms_eden_amax: kernels::ms_eden::amax::module::from_module(module.clone())?,
            ms_eden_fp32: kernels::ms_eden::fp32::module::from_module(module.clone())?,
            ms_eden_fp32_transpose: kernels::ms_eden::fp32_transpose::module::from_module(
                module.clone(),
            )?,
            ms_eden_nvfp4_transpose: kernels::ms_eden::nvfp4_transpose::module::from_module(
                module.clone(),
            )?,
            ms_eden_rowwise_transpose: kernels::ms_eden::rowwise_transpose::from_module(module)?,
        })
    }
}

pub(super) fn validate_csf_scales(
    scales: &cuda_core::DeviceBuffer<u8>,
    rows: u32,
    k: u32,
) -> Result<(), DriverError> {
    if rows == 0
        || k == 0
        || rows % 128 != 0
        || k % 128 != 0
        || scales.len()
            < crate::nvfp4_tma_matmul::scale_layout::sm120_csf_scale_storage_len(
                rows as usize,
                k as usize,
            )
    {
        return Err(DriverError(
            cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
        ));
    }
    Ok(())
}
