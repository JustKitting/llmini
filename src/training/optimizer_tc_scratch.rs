use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4_quant::nvfp4_tensor_amax_chunks;
use rust_kernels_cuda::nvfp4_tma_matmul::kernels::{TILE_K, TILE_M, TILE_N};
use rust_kernels_cuda::nvfp4_tma_matmul::scale_layout::{
    sm120_scale_packed_len, sm120_scale_padded_mn_extent,
};
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;
use rust_kernels_cuda::optimizer::{MUON_COOPERATIVE_BLOCKS, MUON_MATRIX_PHASES};

use super::optimizer_muon::{MUON_MATRIX_SLOTS, max_matrix_dim, max_matrix_len, max_polar_cols};

pub struct MuonScratchBuffers {
    pub(super) oriented: DeviceBuffer<f32>,
    pub(super) polar_next: DeviceBuffer<f32>,
    pub(super) polar_x: DeviceBuffer<f32>,
    pub(super) polar_gram: DeviceBuffer<f32>,
    pub(super) polar_ax: DeviceBuffer<f32>,
    pub(super) polar_chunks: DeviceBuffer<f32>,
    pub(super) tma: MuonTmaScratch,
}

pub(super) struct MuonTmaScratch {
    pub(super) out_padded: DeviceBuffer<f32>,
    pub(super) a: MuonTmaOperandScratch,
    pub(super) b: MuonTmaOperandScratch,
    pub(super) bound_amax: DeviceBuffer<f32>,
    pub(super) descriptors: TmaNvfp4DeviceScaleDescriptors,
}

pub(super) struct MuonTmaOperandScratch {
    pub(super) bytes: DeviceBuffer<u8>,
    pub(super) scales: DeviceBuffer<u8>,
    pub(super) scale_packed: DeviceBuffer<u8>,
    pub(super) global_scale: DeviceBuffer<f32>,
    pub(super) amax: DeviceBuffer<f32>,
    pub(super) chunk_amax: DeviceBuffer<f32>,
}

impl MuonScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            oriented: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_next: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_x: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_gram: DeviceBuffer::zeroed(stream, grouped(max_matrix_dim() * max_matrix_dim()))?,
            polar_ax: DeviceBuffer::zeroed(stream, grouped(max_matrix_len()))?,
            polar_chunks: DeviceBuffer::zeroed(stream, grouped(2 * MUON_COOPERATIVE_BLOCKS))?,
            tma: MuonTmaScratch::new(stream)?,
        })
    }
}

impl MuonTmaScratch {
    fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            out_padded: DeviceBuffer::zeroed(stream, max_tma_out_elements())?,
            a: MuonTmaOperandScratch::new(
                stream,
                max_tma_a_elements(),
                max_tma_a_rows(),
                max_tma_k(),
            )?,
            b: MuonTmaOperandScratch::new(
                stream,
                max_tma_b_elements(),
                max_tma_b_rows(),
                max_tma_k(),
            )?,
            bound_amax: DeviceBuffer::zeroed(stream, 1)?,
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
        })
    }
}

impl MuonTmaOperandScratch {
    fn new(
        stream: &CudaStream,
        elements: usize,
        rows: usize,
        k: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::zeroed(stream, elements / 16)?,
            scale_packed: DeviceBuffer::zeroed(
                stream,
                sm120_scale_packed_len(sm120_scale_padded_mn_extent(rows), k),
            )?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
            amax: DeviceBuffer::zeroed(stream, 1)?,
            chunk_amax: DeviceBuffer::zeroed(stream, nvfp4_tensor_amax_chunks(elements))?,
        })
    }
}

const fn grouped(len: usize) -> usize {
    len * active_matrix_slots()
}

const fn active_matrix_slots() -> usize {
    MUON_MATRIX_SLOTS.div_ceil(MUON_MATRIX_PHASES)
}

const fn max_tma_a_rows() -> usize {
    ceil_to(max_matrix_dim(), TILE_M as usize)
}

const fn max_tma_b_rows() -> usize {
    max2(
        ceil_to(max_matrix_dim(), TILE_N as usize),
        ceil_to(max_polar_cols(), TILE_N as usize),
    )
}

const fn max_tma_k() -> usize {
    ceil_to(max_polar_cols(), TILE_K as usize)
}

const fn max_tma_a_elements() -> usize {
    max_tma_a_rows() * max_tma_k()
}

const fn max_tma_b_elements() -> usize {
    max2(
        ceil_to(max_matrix_dim(), TILE_N as usize) * max_tma_k(),
        ceil_to(max_polar_cols(), TILE_N as usize) * ceil_to(max_matrix_dim(), TILE_K as usize),
    )
}

const fn max_tma_out_elements() -> usize {
    max_tma_a_rows()
        * max2(
            ceil_to(max_matrix_dim(), TILE_N as usize),
            ceil_to(max_polar_cols(), TILE_N as usize),
        )
}

const fn ceil_to(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}
