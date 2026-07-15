use cuda_device::{SharedArray, grid, thread};

use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::float_ptx::max_f32;
use crate::nvfp4_quant::kernels::four_six::helpers::four_six_global_scale;

use super::super::super::super::threads::{WARP_SIZE, WARPS_PER_BLOCK};
use super::super::super::super::work_grid::WorkGrid;
use crate::device_ptr::read_f32;

pub(super) fn reduce_global_scale(
    block_amax: *const f32,
    out_global_scale: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    if work.block() == 0 {
        let tid = thread::threadIdx_x();
        let lane = tid & (WARP_SIZE - 1);
        let warp_in_block = tid / WARP_SIZE;
        reduce_blocks_to_global_scale(
            tid,
            lane,
            warp_in_block,
            block_amax,
            out_global_scale,
            warp_sums,
            work,
        );
    }
    grid::sync();
}

fn reduce_blocks_to_global_scale(
    tid: u32,
    lane: u32,
    warp_in_block: u32,
    block_amax: *const f32,
    out_global_scale: *mut f32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    let mut block = tid;
    let mut local_amax = 0.0;
    while block < work.blocks() {
        local_amax = max_f32(local_amax, read_f32(block_amax, block));
        block += CTA_THREADS;
    }
    let tensor_amax =
        crate::block_reduce::block_max_shared_f32(warp_sums, local_amax, lane, warp_in_block);
    if tid == 0 {
        unsafe {
            *out_global_scale = four_six_global_scale(tensor_amax, 1.0);
        }
    }
}
