use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::cute::Sm120ScaleLayout;
use crate::launch::{grid_x_config, linear_config};

const THREADS_PER_BLOCK: u32 = 256;
const PAIR_K_ATOMS_PER_TILE: u32 = 8;
const PAIR_MN_FAST_PER_TILE: u32 = 8;
const PAIR_MN_FAST_TILES: u32 = Sm120ScaleLayout::MN_FAST / PAIR_MN_FAST_PER_TILE;
const PAIR_TILE_ROWS: usize = (PAIR_MN_FAST_PER_TILE * 4) as usize;
const PAIR_TILE_STRIDE: usize = PAIR_K_ATOMS_PER_TILE as usize + 1;
const PAIR_TILE_WORDS: usize = PAIR_TILE_ROWS * PAIR_TILE_STRIDE;
const QUAD_TILE_WORDS: usize = 4 * PAIR_TILE_WORDS;

#[inline(always)]
fn pair_tile_position(block: u32, k_atom_tiles: u32) -> (u32, u32, u32) {
    let fast_tile = block % PAIR_MN_FAST_TILES;
    let outer = block / PAIR_MN_FAST_TILES;
    let k_atom_tile = outer % k_atom_tiles;
    let mn_block = outer / k_atom_tiles;
    (fast_tile * PAIR_MN_FAST_PER_TILE, k_atom_tile, mn_block)
}

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

    #[kernel]
    pub fn pack_sm120_scale_plane_pow2_kernel(
        logical: &[u8],
        mut packed: DisjointSlice<u8>,
        mn_extent: u32,
        padded_mn_extent: u32,
        k_atoms: u32,
        k_atoms_shift: u32,
    ) {
        let stride = thread::blockDim_x() * thread::gridDim_x();
        let mut index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let packed_words = padded_mn_extent * k_atoms;
        let k_dim = k_atoms * Sm120ScaleLayout::K_ATOM;
        while index < packed_words {
            let mn = index >> k_atoms_shift;
            let k_atom = index & (k_atoms - 1);
            let value = if mn < mn_extent {
                let src = (index * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
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

    #[kernel]
    pub fn pack_sm120_scale_plane_pair_tiled_kernel(
        logical_0: &[u8],
        mut packed_0: DisjointSlice<u8>,
        mn_extent_0: u32,
        logical_1: &[u8],
        mut packed_1: DisjointSlice<u8>,
        mn_extent_1: u32,
        k_atoms: u32,
        k_atom_tiles: u32,
        blocks_0: u32,
        blocks_1: u32,
    ) {
        static mut TILES: SharedArray<u32, { 2 * PAIR_TILE_WORDS }> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let fast_tile = block % PAIR_MN_FAST_TILES;
        let outer = block / PAIR_MN_FAST_TILES;
        let k_atom_tile = outer % k_atom_tiles;
        let mn_block = outer / k_atom_tiles;
        let fast_base = fast_tile * PAIR_MN_FAST_PER_TILE;
        let thread_id = thread::threadIdx_x();
        let warp = thread_id / 32;
        let load_row = warp + (thread_id % 32 / PAIR_K_ATOMS_PER_TILE) * 8;
        let load_k = thread_id % PAIR_K_ATOMS_PER_TILE;
        let mn_fast = fast_base + load_row / 4;
        let mn_slow = load_row % 4;
        let mn =
            mn_block * Sm120ScaleLayout::MN_BLOCK + mn_slow * Sm120ScaleLayout::MN_FAST + mn_fast;
        let k_atom = k_atom_tile * PAIR_K_ATOMS_PER_TILE + load_k;
        let value_0 = if block < blocks_0 && mn < mn_extent_0 && k_atom < k_atoms {
            let src = ((mn * k_atoms + k_atom) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_0.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };
        let value_1 = if block < blocks_1 && mn < mn_extent_1 && k_atom < k_atoms {
            let src = ((mn * k_atoms + k_atom) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_1.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };
        let shared = unsafe { (&mut *(&raw mut TILES)).as_mut_ptr() };
        let tile_index = load_row as usize * PAIR_TILE_STRIDE + load_k as usize;
        unsafe {
            *shared.add(tile_index) = value_0;
            *shared.add(PAIR_TILE_WORDS + tile_index) = value_1;
        }
        thread::sync_threads();

        let store_k = thread_id / PAIR_TILE_ROWS as u32;
        let store_row = thread_id % PAIR_TILE_ROWS as u32;
        let k_atom = k_atom_tile * PAIR_K_ATOMS_PER_TILE + store_k;
        if k_atom < k_atoms {
            let store_mn_fast = fast_base + store_row / 4;
            let store_mn_slow = store_row % 4;
            let store_mn = mn_block * Sm120ScaleLayout::MN_BLOCK
                + store_mn_slow * Sm120ScaleLayout::MN_FAST
                + store_mn_fast;
            let dst_word = Sm120ScaleLayout::block_major_u32_word_offset(store_mn, k_atom, k_atoms);
            let tile_index = store_row as usize * PAIR_TILE_STRIDE + store_k as usize;
            if block < blocks_0 {
                let value = unsafe { *shared.add(tile_index) };
                unsafe {
                    *((packed_0.as_mut_ptr() as *mut u32).add(dst_word)) = value;
                }
            }
            if block < blocks_1 {
                let value = unsafe { *shared.add(PAIR_TILE_WORDS + tile_index) };
                unsafe {
                    *((packed_1.as_mut_ptr() as *mut u32).add(dst_word)) = value;
                }
            }
        }
    }

    #[kernel]
    pub fn pack_sm120_scale_plane_quad_tiled_kernel(
        logical_0: &[u8],
        mut packed_0: DisjointSlice<u8>,
        mn_extent_0: u32,
        logical_1: &[u8],
        mut packed_1: DisjointSlice<u8>,
        mn_extent_1: u32,
        logical_2: &[u8],
        mut packed_2: DisjointSlice<u8>,
        mn_extent_2: u32,
        logical_3: &[u8],
        mut packed_3: DisjointSlice<u8>,
        mn_extent_3: u32,
        k_atoms_01: u32,
        k_atom_tiles_01: u32,
        k_atoms_23: u32,
        k_atom_tiles_23: u32,
        blocks_0: u32,
        blocks_1: u32,
        blocks_2: u32,
        blocks_3: u32,
    ) {
        static mut TILES: SharedArray<u32, QUAD_TILE_WORDS> = SharedArray::UNINIT;

        let block = thread::blockIdx_x();
        let (fast_base_01, k_atom_tile_01, mn_block_01) =
            pair_tile_position(block, k_atom_tiles_01);
        let (fast_base_23, k_atom_tile_23, mn_block_23) =
            pair_tile_position(block, k_atom_tiles_23);
        let thread_id = thread::threadIdx_x();
        let warp = thread_id / 32;
        let load_row = warp + (thread_id % 32 / PAIR_K_ATOMS_PER_TILE) * 8;
        let load_k = thread_id % PAIR_K_ATOMS_PER_TILE;

        let mn_fast_01 = fast_base_01 + load_row / 4;
        let mn_slow_01 = load_row % 4;
        let mn_01 = mn_block_01 * Sm120ScaleLayout::MN_BLOCK
            + mn_slow_01 * Sm120ScaleLayout::MN_FAST
            + mn_fast_01;
        let k_atom_01 = k_atom_tile_01 * PAIR_K_ATOMS_PER_TILE + load_k;
        let value_0 = if block < blocks_0 && mn_01 < mn_extent_0 && k_atom_01 < k_atoms_01 {
            let src =
                ((mn_01 * k_atoms_01 + k_atom_01) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_0.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };
        let value_1 = if block < blocks_1 && mn_01 < mn_extent_1 && k_atom_01 < k_atoms_01 {
            let src =
                ((mn_01 * k_atoms_01 + k_atom_01) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_1.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };

        let mn_fast_23 = fast_base_23 + load_row / 4;
        let mn_slow_23 = load_row % 4;
        let mn_23 = mn_block_23 * Sm120ScaleLayout::MN_BLOCK
            + mn_slow_23 * Sm120ScaleLayout::MN_FAST
            + mn_fast_23;
        let k_atom_23 = k_atom_tile_23 * PAIR_K_ATOMS_PER_TILE + load_k;
        let value_2 = if block < blocks_2 && mn_23 < mn_extent_2 && k_atom_23 < k_atoms_23 {
            let src =
                ((mn_23 * k_atoms_23 + k_atom_23) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_2.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };
        let value_3 = if block < blocks_3 && mn_23 < mn_extent_3 && k_atom_23 < k_atoms_23 {
            let src =
                ((mn_23 * k_atoms_23 + k_atom_23) * Sm120ScaleLayout::GROUPS_PER_K_ATOM) as usize;
            unsafe { *(logical_3.as_ptr().add(src) as *const u32) }
        } else {
            u32::from_le_bytes([Sm120ScaleLayout::PAD_BYTE; 4])
        };

        let shared = unsafe { (&mut *(&raw mut TILES)).as_mut_ptr() };
        let tile_index = load_row as usize * PAIR_TILE_STRIDE + load_k as usize;
        unsafe {
            *shared.add(tile_index) = value_0;
            *shared.add(PAIR_TILE_WORDS + tile_index) = value_1;
            *shared.add(2 * PAIR_TILE_WORDS + tile_index) = value_2;
            *shared.add(3 * PAIR_TILE_WORDS + tile_index) = value_3;
        }
        thread::sync_threads();

        let store_k = thread_id / PAIR_TILE_ROWS as u32;
        let store_row = thread_id % PAIR_TILE_ROWS as u32;
        let store_k_atom_01 = k_atom_tile_01 * PAIR_K_ATOMS_PER_TILE + store_k;
        if store_k_atom_01 < k_atoms_01 {
            let store_mn_fast = fast_base_01 + store_row / 4;
            let store_mn_slow = store_row % 4;
            let store_mn = mn_block_01 * Sm120ScaleLayout::MN_BLOCK
                + store_mn_slow * Sm120ScaleLayout::MN_FAST
                + store_mn_fast;
            let dst_word = Sm120ScaleLayout::block_major_u32_word_offset(
                store_mn,
                store_k_atom_01,
                k_atoms_01,
            );
            let tile_index = store_row as usize * PAIR_TILE_STRIDE + store_k as usize;
            if block < blocks_0 {
                unsafe {
                    *((packed_0.as_mut_ptr() as *mut u32).add(dst_word)) = *shared.add(tile_index);
                }
            }
            if block < blocks_1 {
                unsafe {
                    *((packed_1.as_mut_ptr() as *mut u32).add(dst_word)) =
                        *shared.add(PAIR_TILE_WORDS + tile_index);
                }
            }
        }

        let store_k_atom_23 = k_atom_tile_23 * PAIR_K_ATOMS_PER_TILE + store_k;
        if store_k_atom_23 < k_atoms_23 {
            let store_mn_fast = fast_base_23 + store_row / 4;
            let store_mn_slow = store_row % 4;
            let store_mn = mn_block_23 * Sm120ScaleLayout::MN_BLOCK
                + store_mn_slow * Sm120ScaleLayout::MN_FAST
                + store_mn_fast;
            let dst_word = Sm120ScaleLayout::block_major_u32_word_offset(
                store_mn,
                store_k_atom_23,
                k_atoms_23,
            );
            let tile_index = store_row as usize * PAIR_TILE_STRIDE + store_k as usize;
            if block < blocks_2 {
                unsafe {
                    *((packed_2.as_mut_ptr() as *mut u32).add(dst_word)) =
                        *shared.add(2 * PAIR_TILE_WORDS + tile_index);
                }
            }
            if block < blocks_3 {
                unsafe {
                    *((packed_3.as_mut_ptr() as *mut u32).add(dst_word)) =
                        *shared.add(3 * PAIR_TILE_WORDS + tile_index);
                }
            }
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
        if k_atoms.is_power_of_two() {
            self.module.pack_sm120_scale_plane_pow2_kernel(
                stream,
                linear_config(active, THREADS_PER_BLOCK),
                logical,
                packed,
                mn_extent,
                padded_mn_extent,
                k_atoms,
                k_atoms.trailing_zeros(),
            )
        } else {
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

    #[allow(clippy::too_many_arguments)]
    pub fn pack_pair(
        &self,
        stream: &CudaStream,
        logical_0: &DeviceBuffer<u8>,
        packed_0: &mut DeviceBuffer<u8>,
        mn_extent_0: u32,
        logical_1: &DeviceBuffer<u8>,
        packed_1: &mut DeviceBuffer<u8>,
        mn_extent_1: u32,
        k_dim: u32,
    ) -> Result<(), DriverError> {
        assert!(k_dim.is_multiple_of(Sm120ScaleLayout::VECTOR_SIZE));
        let k_groups = Sm120ScaleLayout::k_groups(k_dim);
        assert!(k_groups.is_multiple_of(Sm120ScaleLayout::GROUPS_PER_K_ATOM));
        let k_atoms = k_groups / Sm120ScaleLayout::GROUPS_PER_K_ATOM;
        let padded_mn_extent_0 = Sm120ScaleLayout::padded_mn_extent(mn_extent_0);
        let padded_mn_extent_1 = Sm120ScaleLayout::padded_mn_extent(mn_extent_1);
        let k_atom_tiles = k_atoms.div_ceil(PAIR_K_ATOMS_PER_TILE);
        let blocks_0 =
            padded_mn_extent_0 / Sm120ScaleLayout::MN_BLOCK * k_atom_tiles * PAIR_MN_FAST_TILES;
        let blocks_1 =
            padded_mn_extent_1 / Sm120ScaleLayout::MN_BLOCK * k_atom_tiles * PAIR_MN_FAST_TILES;
        assert!(logical_0.len() >= (mn_extent_0 * k_groups) as usize);
        assert!(logical_1.len() >= (mn_extent_1 * k_groups) as usize);
        assert!(packed_0.len() >= Sm120ScaleLayout::packed_len(padded_mn_extent_0, k_dim));
        assert!(packed_1.len() >= Sm120ScaleLayout::packed_len(padded_mn_extent_1, k_dim));
        self.module.pack_sm120_scale_plane_pair_tiled_kernel(
            stream,
            grid_x_config(blocks_0.max(blocks_1), THREADS_PER_BLOCK),
            logical_0,
            packed_0,
            mn_extent_0,
            logical_1,
            packed_1,
            mn_extent_1,
            k_atoms,
            k_atom_tiles,
            blocks_0,
            blocks_1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn pack_quad(
        &self,
        stream: &CudaStream,
        logical_0: &DeviceBuffer<u8>,
        packed_0: &mut DeviceBuffer<u8>,
        mn_extent_0: u32,
        logical_1: &DeviceBuffer<u8>,
        packed_1: &mut DeviceBuffer<u8>,
        mn_extent_1: u32,
        k_dim_01: u32,
        logical_2: &DeviceBuffer<u8>,
        packed_2: &mut DeviceBuffer<u8>,
        mn_extent_2: u32,
        logical_3: &DeviceBuffer<u8>,
        packed_3: &mut DeviceBuffer<u8>,
        mn_extent_3: u32,
        k_dim_23: u32,
    ) -> Result<(), DriverError> {
        let (k_atoms_01, k_atom_tiles_01, blocks_0, blocks_1) = quad_pair_shape(
            logical_0,
            packed_0,
            mn_extent_0,
            logical_1,
            packed_1,
            mn_extent_1,
            k_dim_01,
        );
        let (k_atoms_23, k_atom_tiles_23, blocks_2, blocks_3) = quad_pair_shape(
            logical_2,
            packed_2,
            mn_extent_2,
            logical_3,
            packed_3,
            mn_extent_3,
            k_dim_23,
        );
        self.module.pack_sm120_scale_plane_quad_tiled_kernel(
            stream,
            grid_x_config(
                blocks_0.max(blocks_1).max(blocks_2).max(blocks_3),
                THREADS_PER_BLOCK,
            ),
            logical_0,
            packed_0,
            mn_extent_0,
            logical_1,
            packed_1,
            mn_extent_1,
            logical_2,
            packed_2,
            mn_extent_2,
            logical_3,
            packed_3,
            mn_extent_3,
            k_atoms_01,
            k_atom_tiles_01,
            k_atoms_23,
            k_atom_tiles_23,
            blocks_0,
            blocks_1,
            blocks_2,
            blocks_3,
        )
    }
}

fn quad_pair_shape(
    logical_0: &DeviceBuffer<u8>,
    packed_0: &DeviceBuffer<u8>,
    mn_extent_0: u32,
    logical_1: &DeviceBuffer<u8>,
    packed_1: &DeviceBuffer<u8>,
    mn_extent_1: u32,
    k_dim: u32,
) -> (u32, u32, u32, u32) {
    assert!(k_dim.is_multiple_of(Sm120ScaleLayout::VECTOR_SIZE));
    let k_groups = Sm120ScaleLayout::k_groups(k_dim);
    assert!(k_groups.is_multiple_of(Sm120ScaleLayout::GROUPS_PER_K_ATOM));
    let k_atoms = k_groups / Sm120ScaleLayout::GROUPS_PER_K_ATOM;
    let k_atom_tiles = k_atoms.div_ceil(PAIR_K_ATOMS_PER_TILE);
    let padded_0 = Sm120ScaleLayout::padded_mn_extent(mn_extent_0);
    let padded_1 = Sm120ScaleLayout::padded_mn_extent(mn_extent_1);
    let blocks_0 = padded_0 / Sm120ScaleLayout::MN_BLOCK * k_atom_tiles * PAIR_MN_FAST_TILES;
    let blocks_1 = padded_1 / Sm120ScaleLayout::MN_BLOCK * k_atom_tiles * PAIR_MN_FAST_TILES;
    assert!(logical_0.len() >= (mn_extent_0 * k_groups) as usize);
    assert!(logical_1.len() >= (mn_extent_1 * k_groups) as usize);
    assert!(packed_0.len() >= Sm120ScaleLayout::packed_len(padded_0, k_dim));
    assert!(packed_1.len() >= Sm120ScaleLayout::packed_len(padded_1, k_dim));
    (k_atoms, k_atom_tiles, blocks_0, blocks_1)
}
