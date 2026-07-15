use cuda_device::{SharedArray, thread};

use crate::float_ptx::max_f32;
use crate::float_ptx::sqrt_f32;

use super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::work_grid::WorkGrid;

mod chunk;
mod one;

const UPDATE_VALUES_PER_CHUNK: u32 = crate::nvfp4_quant::NVFP4_TENSOR_AMAX_VALUES_PER_BLOCK as u32;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn update_master_chunks(
    u: *const f32,
    z_master: *mut f32,
    x_master: *mut f32,
    block_amax: *mut f32,
    rows: u32,
    cols: u32,
    len: u32,
    transposed: bool,
    learning_rate: f32,
    weight_decay: f32,
    average_coefficient: f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let tid = thread::threadIdx_x();
    let lane = tid & (WARP_SIZE - 1);
    let warp_in_block = tid / WARP_SIZE;
    let mut chunk = work.block();
    let chunk_count = len.div_ceil(UPDATE_VALUES_PER_CHUNK);
    let mut local_block_amax = 0.0;
    let scale = 0.2 * sqrt_f32(max_f32(rows as f32, cols as f32));

    while chunk < chunk_count {
        let base = chunk * UPDATE_VALUES_PER_CHUNK;
        let local_amax = chunk::update_eight_amax(
            u,
            z_master,
            x_master,
            rows,
            cols,
            len,
            transposed,
            scale,
            learning_rate,
            weight_decay,
            average_coefficient,
            base,
            tid,
        );
        let block_amax =
            crate::block_reduce::block_max_shared_f32(warp_sums, local_amax, lane, warp_in_block);
        local_block_amax = max_f32(local_block_amax, block_amax);
        chunk += work.blocks();
    }
    if tid == 0 {
        unsafe {
            *block_amax.add(work.block() as usize) = local_block_amax;
        }
    }
}
