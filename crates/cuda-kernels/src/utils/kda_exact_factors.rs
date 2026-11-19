//! Components for complete bounded-interval KDA, loaded only by explicit users.
//! No default runtime dispatch or training graph is selected by this module.
use cuda_core::{CudaModule, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel};
use std::sync::Arc;

pub struct KdaExactFactorModule {
    kernels: kernels::LoadedModule,
}

pub struct FormedInverseArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub lower: &'a DeviceBuffer<f32>,
    pub out: &'out mut DeviceBuffer<u16>,
    pub chunk_batch: u32,
}

impl KdaExactFactorModule {
    pub fn from_module(module: Arc<CudaModule>) -> Result<Self, DriverError> {
        if !crate::module_exports::contains(&module, b"kda_exact_formed_inverse_kernel")? {
            return Err(DriverError(1));
        }
        Ok(Self {
            kernels: kernels::from_module(module)?,
        })
    }

    pub fn formed_inverse(&self, args: FormedInverseArgs<'_, '_>) -> Result<(), DriverError> {
        let Some(count) = args.chunk_batch.checked_mul(64 * 64).filter(|&n| n > 0) else {
            return Err(DriverError(1));
        };
        if args.lower.len() < count as usize || args.out.len() < count as usize {
            return Err(DriverError(1));
        }
        self.kernels.kda_exact_formed_inverse_kernel(
            args.stream,
            LaunchConfig {
                grid_dim: (args.chunk_batch, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            args.lower,
            args.out,
            args.chunk_batch,
        )
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn kda_exact_formed_inverse_kernel(
        lower: &[f32],
        out: DisjointSlice<u16>,
        chunk_batch: u32,
    ) {
        static mut RAW: SharedArray<f32, { crate::f16_tc_matmul::KDA_INVERSE_SHARED_ELEMS }> =
            SharedArray::UNINIT;
        unsafe {
            crate::f16_tc_matmul::formed_inverse_body(lower, out, &mut RAW, chunk_batch);
        }
    }
}
