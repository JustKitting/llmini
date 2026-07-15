use cuda_device::{SharedArray, thread};

use super::{NEG_INFINITY, WARPS_PER_BLOCK};
use crate::warp_reduce::{warp_max_finite_f32, warp_sum_f32};

pub(super) fn block_reduce_max(
    local: f32,
    tid: u32,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let lane = tid & 31;
    let warp = tid / 32;
    let warp_value = warp_max_finite_f32(local);
    if lane == 0 {
        reduce[warp as usize] = warp_value;
    }
    thread::sync_threads();

    let partial = if warp == 0 && lane < WARPS_PER_BLOCK as u32 {
        reduce[lane as usize]
    } else {
        NEG_INFINITY
    };
    let block = warp_max_finite_f32(partial);
    if warp == 0 && lane == 0 {
        reduce[0] = block;
    }
    thread::sync_threads();
    reduce[0]
}

pub(super) fn block_reduce_sum(
    local: f32,
    tid: u32,
    reduce: &mut SharedArray<f32, WARPS_PER_BLOCK>,
) -> f32 {
    let lane = tid & 31;
    let warp = tid / 32;
    let warp_value = warp_sum_f32(local);
    if lane == 0 {
        reduce[warp as usize] = warp_value;
    }
    thread::sync_threads();

    let mut block = 0.0;
    if tid == 0 {
        let mut i = 0;
        while i < WARPS_PER_BLOCK {
            block += reduce[i];
            i += 1;
        }
        reduce[0] = block;
    }
    thread::sync_threads();
    reduce[0]
}
