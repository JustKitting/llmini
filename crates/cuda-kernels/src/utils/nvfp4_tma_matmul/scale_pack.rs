use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

use super::cute::Sm120ScaleLayout;
use crate::launch::linear_config;

const THREADS_PER_BLOCK: u32 = 256;

#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn pack_sm120_scale_plane_kernel(
        logical: &[u8],
        mut packed: DisjointSlice<u8>,
        mn_extent: u32,
        padded_mn_extent: u32,
        k_atoms: u32,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let packed_words = padded_mn_extent * k_atoms;
        let k_dim = k_atoms * Sm120ScaleLayout::K_ATOM;
        while index < packed_words {
            let mn = index / k_atoms;
            let k_atom = index - mn * k_atoms;
            let value = if mn < mn_extent {
                let src = ((mn * k_atoms + k_atom) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
                unsafe { *(logical.as_ptr().add(src) as *const u32) }
            } else {
                u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
            };
            let dst = Sm120ScaleLayout::block_major_byte_offset(
                mn,
                k_atom * Sm120ScaleLayout::GROUPS_PER_K_ATOM,
                padded_mn_extent,
                k_dim,
            );
            unsafe {
                *(packed.as_mut_ptr().add(dst) as *mut u32) = value;
            }
            index += stride;
        }
    }
}

pub struct Sm120ScalePackModule {
    module: module::LoadedModule,
}

impl Sm120ScalePackModule {
    pub fn from_module(module: std::sync::Arc<cuda_core::CudaModule>) -> Result<Self, DriverError> {
        Ok(Self {
            module: module::from_module(module)?,
        })
    }

    pub fn pack(
        &self,
        stream: &CudaStream,
        logical: &DeviceBuffer<u8>,
        packed: &mut DeviceBuffer<u8>,
        mn_extent: u32,
        k_dim: u32,
    ) -> Result<(), DriverError> {
        assert!(k_dim.is_multiple_of(Sm120ScaleLayout::VECTOR_SIZE));
        let k_groups = Sm120ScaleLayout::k_groups(k_dim);
        assert!(k_groups.is_multiple_of(Sm120ScaleLayout::GROUPS_PER_K_ATOM));
        let k_atoms = k_groups / Sm120ScaleLayout::GROUPS_PER_K_ATOM;
        let padded_mn_extent = Sm120ScaleLayout::padded_mn_extent(mn_extent);
        let active = padded_mn_extent * k_atoms;
        assert!(logical.len() >= (mn_extent * k_groups) as usize);
        assert!(packed.len() >= Sm120ScaleLayout::packed_len(padded_mn_extent, k_dim));
        self.module.pack_sm120_scale_plane_kernel(
            stream,
            linear_config(active, THREADS_PER_BLOCK),
            logical,
            packed,
            mn_extent,
            padded_mn_extent,
            k_atoms,
        )
    }
}
