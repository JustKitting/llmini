use cuda_device::{DisjointSlice, SharedArray, thread};

use crate::block_reduce::block_sum_shared_f32;
use crate::device_ptr::write_f32;
use crate::float_ptx::sqrt_f32;
use crate::warp_reduce::thread_lane_warp;

use super::{THREADS_PER_BLOCK, WARP_SUM_SLOTS};

pub(super) fn grad_clip_scale_body(
    chunk_sums: &[f32],
    extra_sums: &[f32],
    scale: &mut DisjointSlice<f32>,
    norm_out: &mut DisjointSlice<f32>,
    chunk_count: u32,
    extra_sum_count: u32,
    max_norm: f32,
) {
    static mut WARP_SUMS: SharedArray<f32, WARP_SUM_SLOTS> = SharedArray::UNINIT;

    let (thread, lane, warp) = thread_lane_warp();
    let mut local = 0.0;
    let mut chunk = thread;

    while chunk < chunk_count {
        local += chunk_sums[chunk as usize];
        chunk += THREADS_PER_BLOCK;
    }
    let mut extra = thread;
    while extra < extra_sum_count {
        local += extra_sums[extra as usize];
        extra += THREADS_PER_BLOCK;
    }

    let sum = unsafe { block_sum_shared_f32(&mut WARP_SUMS, local, lane, warp) };
    if thread::threadIdx_x() == 0 {
        let norm = sqrt_f32(sum);
        let value = if norm > max_norm {
            max_norm / (norm + 1.0e-6)
        } else {
            1.0
        };
        write_f32(scale.as_mut_ptr(), 0, value);
        write_f32(norm_out.as_mut_ptr(), 0, norm);
    }
}
