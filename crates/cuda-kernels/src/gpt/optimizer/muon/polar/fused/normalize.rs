use super::super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::super::work_grid::WorkGrid;
use crate::device_ptr::{read_f32, write_f32};
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::{fma_f32, sqrt_f32};
use cuda_device::{SharedArray, grid, thread};

const POLAR_EXPRESS_NORM_SAFETY: f32 = 1.01;
const POLAR_EXPRESS_EPS: f32 = 1.0e-7;

#[derive(Clone, Copy)]
struct NormalizeSourceToX {
    source: *const f32,
    x: *mut f32,
    chunks: *mut f32,
    source_rows: u32,
    source_cols: u32,
    polar_rows: u32,
    polar_cols: u32,
    transpose_source: bool,
}

impl NormalizeSourceToX {
    #[inline(always)]
    fn source_index(self, polar_index: u32) -> u32 {
        if !self.transpose_source {
            return polar_index;
        }
        let row = polar_index / self.source_rows;
        let col = polar_index - row * self.source_rows;
        col * self.source_cols + row
    }
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(crate) fn normalize_source_to_x(
    source: *const f32,
    x: *mut f32,
    chunks: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
    source_rows: u32,
    source_cols: u32,
    polar_rows: u32,
    polar_cols: u32,
    transpose_source: bool,
) {
    let job = NormalizeSourceToX {
        source,
        x,
        chunks,
        source_rows,
        source_cols,
        polar_rows,
        polar_cols,
        transpose_source,
    };
    let len = job.source_rows * job.source_cols;
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp_in_block = tid / WARP_SIZE;
    let stride = work.stride();
    let mut local = 0.0;
    let mut index = work.thread();

    while index < len {
        let value = read_f32(job.source, index);
        local = fma_f32(value, value, local);
        index += stride;
    }

    let local_sum =
        crate::block_reduce::block_sum_shared_f32(warp_sums, local, lane, warp_in_block);
    if tid == 0 {
        write_f32(job.chunks, work.block(), local_sum);
    }
    grid::sync();

    normalize_source_to_x_from_chunks(job, warp_sums, work);
}

fn normalize_source_to_x_from_chunks(
    job: NormalizeSourceToX,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp_in_block = tid / WARP_SIZE;
    let stride = work.stride();

    if work.block() == 0 {
        let mut local = 0.0;
        let mut chunk = tid;
        while chunk < work.blocks() {
            local += read_f32(job.chunks, chunk);
            chunk += CTA_THREADS;
        }
        let sum = crate::block_reduce::block_sum_shared_f32(warp_sums, local, lane, warp_in_block);
        if tid == 0 {
            write_f32(
                job.chunks,
                0,
                1.0 / (sqrt_f32(sum) * POLAR_EXPRESS_NORM_SAFETY + POLAR_EXPRESS_EPS),
            );
        }
    }
    grid::sync();

    let inv_norm = read_f32(job.chunks, 0);
    let polar_len = job.polar_rows * job.polar_cols;
    let mut polar_index = work.thread();
    while polar_index < polar_len {
        let src = job.source_index(polar_index);
        write_f32(job.x, polar_index, read_f32(job.source, src) * inv_norm);
        polar_index += stride;
    }
}
