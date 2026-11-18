//! Fused existing8192-wide mHC RMS and native Four-Six FP4/CSF producer.
//! Keep the original sum and multiply rounding. No global normalized FP32
//! output or extra state tape; the backward path still uses the saved inverse.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};
use std::sync::Arc;
pub struct MhcRmsQuantModule {
    kernels: kernels::LoadedModule,
}
pub struct RmsQuantArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub input: &'a DeviceBuffer<f32>,
    pub inverse: &'out mut DeviceBuffer<f32>,
    pub amax: &'out mut DeviceBuffer<f32>,
    pub payload: &'out mut DeviceBuffer<u8>,
    pub scales: &'out mut DeviceBuffer<u8>,
    pub global_scales: &'out mut DeviceBuffer<f32>,
    pub rows: u32,
}
impl MhcRmsQuantModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"mhc_rms_quant_8192_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }
    pub fn forward(&self, a: RmsQuantArgs<'_, '_>) -> Result<(), DriverError> {
        let n = a
            .rows
            .checked_mul(8192)
            .filter(|&n| n > 0)
            .ok_or(DriverError(1))? as usize;
        let scale_count = crate::nvfp4::csf::rowwise_encoded_len(a.rows as usize, 8192);
        if a.input.len() < n
            || a.payload.len() < n / 2
            || a.scales.len() < scale_count
            || [a.inverse.len(), a.amax.len(), a.global_scales.len()]
                .iter()
                .any(|&len| len < a.rows as usize)
        {
            return Err(DriverError(1));
        }
        self.kernels.mhc_rms_quant_8192_kernel(
            a.stream,
            LaunchConfig {
                grid_dim: (a.rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            a.input,
            a.inverse,
            a.amax,
            a.payload,
            a.scales,
            a.global_scales,
            a.rows,
        )
    }
}
#[cuda_module]
pub mod kernels {
    use super::*;
    #[kernel]
    pub fn mhc_rms_quant_8192_kernel(
        input: &[f32],
        inverse: DisjointSlice<f32>,
        amax: DisjointSlice<f32>,
        payload: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        global: DisjointSlice<f32>,
        rows: u32,
    ) {
        static mut REDUCE: SharedArray<f32, 8> = SharedArray::UNINIT;
        unsafe {
            crate::nvfp4_quant::kernels::four_six::module::mhc_rms_quant_body(
                input,
                inverse,
                amax,
                payload,
                scales,
                global,
                rows,
                &mut REDUCE,
            );
        }
    }
}
