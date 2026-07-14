use cuda_core::LaunchConfig;

use super::super::config::{GROUPS_PER_BLOCK, THREADS_PER_BLOCK, WARPS_PER_BLOCK};
use super::super::kernels;
use crate::launch::grid_x_config;

const MS_EDEN_CHUNK_LEN: u32 = 32;

#[derive(Clone, Copy)]
pub(in crate::nvfp4_quant) struct MsEdenPackGrid {
    pub chunk_count: u32,
    pub grid_dim: u32,
}

impl MsEdenPackGrid {
    pub fn for_elements(element_count: u32) -> Self {
        Self::from_chunks(element_count.div_ceil(MS_EDEN_CHUNK_LEN))
    }

    fn from_chunks(chunk_count: u32) -> Self {
        Self {
            chunk_count,
            grid_dim: chunk_count.div_ceil(WARPS_PER_BLOCK),
        }
    }

    pub fn config(self) -> LaunchConfig {
        grid_config(self.grid_dim)
    }

    pub fn is_exact(self) -> bool {
        self.chunk_count.is_multiple_of(WARPS_PER_BLOCK)
    }
}

pub(in crate::nvfp4_quant) fn grid_config(grid_x: u32) -> LaunchConfig {
    grid_x_config(grid_x, THREADS_PER_BLOCK)
}

pub(in crate::nvfp4_quant) fn four_six_grid_config(group_count: u32) -> LaunchConfig {
    grid_config(group_count.div_ceil(GROUPS_PER_BLOCK))
}

pub(in crate::nvfp4_quant) fn four_six_transpose_tiled_config(
    source_rows: u32,
    source_cols: u32,
) -> LaunchConfig {
    crate::launch::launch_config((source_cols / 64, source_rows / 16, 1), THREADS_PER_BLOCK)
}

pub(in crate::nvfp4_quant) fn ms_eden_rowwise_transpose_tiled_config(
    source_rows: u32,
    source_cols: u32,
) -> LaunchConfig {
    crate::launch::launch_config((source_rows / 32, source_cols / 8, 1), THREADS_PER_BLOCK)
}

pub(in crate::nvfp4_quant) fn four_six_rowwise_pow2(row_len: u32, group_count: u32) -> bool {
    row_len.is_power_of_two() && group_count.is_multiple_of(GROUPS_PER_BLOCK)
}

pub(in crate::nvfp4_quant) fn tensor_amax_chunk_count(element_count: u32) -> u32 {
    element_count.div_ceil(kernels::row_amax::TENSOR_AMAX_VALUES_PER_BLOCK)
}

pub(super) fn ms_eden_chunks(row_len: u32) -> Option<u32> {
    row_len
        .is_multiple_of(MS_EDEN_CHUNK_LEN)
        .then_some(row_len / MS_EDEN_CHUNK_LEN)
}
