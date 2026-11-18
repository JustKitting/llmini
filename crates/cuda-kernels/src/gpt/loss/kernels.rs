use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use super::{CROSS_ENTROPY_THREADS_PER_BLOCK, CROSS_ENTROPY_WARPS_PER_BLOCK, CrossEntropyParams};
use crate::block_reduce::{
    block_max_leader_f32, block_max_shared_f32_for_warps, block_sum_shared_f32,
};
use crate::float_ptx::{abs_f32, exp_preserve_subnormal_f32, log1p_f32, max_f32};
use crate::warp_reduce::thread_lane_warp;

pub use module::{LoadedModule, from_module};

#[cuda_module]
mod module {
    use super::*;

    #[kernel]
    pub fn cross_entropy_kernel(
        logits: &[f32],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        mut dlogits: DisjointSlice<f32>,
        mut dlogits_row_amax: DisjointSlice<f32>,
        params: CrossEntropyParams,
    ) {
        static mut REDUCE: SharedArray<f32, { CROSS_ENTROPY_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let (thread, lane, warp_in_block) = thread_lane_warp();

        if row < params.token_count {
            let row_base = row as usize * params.vocab_size as usize;
            let mut local_max = f32::NEG_INFINITY;
            let mut col = thread;
            while col < params.vocab_size {
                local_max = max_f32(local_max, logits[row_base + col as usize]);
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }
            let row_max = unsafe {
                block_max_shared_f32_for_warps(
                    &mut REDUCE,
                    CROSS_ENTROPY_WARPS_PER_BLOCK,
                    local_max,
                    lane,
                    warp_in_block,
                    f32::NEG_INFINITY,
                )
            };
            // Exclude exactly one maximum, retaining all tied classes.
            let mut local_argmax = f32::NEG_INFINITY;
            col = thread;
            while col < params.vocab_size {
                if logits[row_base + col as usize] == row_max {
                    local_argmax = max_f32(local_argmax, -(col as f32));
                }
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }
            let maximum_col = -unsafe {
                block_max_shared_f32_for_warps(&mut REDUCE,CROSS_ENTROPY_WARPS_PER_BLOCK,
                    local_argmax,lane,warp_in_block,f32::NEG_INFINITY)
            } as i32;
            let mut local_sum = 0.0_f32;
            col = thread;
            while col < params.vocab_size {
                if col != maximum_col as u32 {
                    local_sum += exp_preserve_subnormal_f32(logits[row_base + col as usize] - row_max);
                }
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }
            let nonmaximum_sum = unsafe {
                block_sum_shared_f32(&mut REDUCE, local_sum, lane, warp_in_block)
            };
            let denom = 1.0 + nonmaximum_sum;
            let target = targets[row as usize];
            if thread == 0 {
                let target_logit = logits[row_base + target as usize];
                unsafe {
                    *losses.get_unchecked_mut(row as usize) =
                        (row_max - target_logit) + log1p_f32(nonmaximum_sum);
                }
            }

            let mut local_dlogits_amax = 0.0_f32;
            let mut local_nontarget_sum = 0.0_f32;
            col = thread;
            while col < params.vocab_size {
                if col != target {
                    let probability = exp_preserve_subnormal_f32(logits[row_base + col as usize] - row_max) / denom;
                    let dlogit = probability / params.token_count as f32;
                    local_nontarget_sum += dlogit;
                    local_dlogits_amax = max_f32(local_dlogits_amax, abs_f32(dlogit));
                    unsafe { *dlogits.get_unchecked_mut(row_base + col as usize) = dlogit; }
                }
                col += CROSS_ENTROPY_THREADS_PER_BLOCK;
            }

            let target_gradient = -unsafe {
                block_sum_shared_f32(&mut REDUCE, local_nontarget_sum, lane, warp_in_block)
            };
            if thread == 0 {
                unsafe { *dlogits.get_unchecked_mut(row_base + target as usize) = target_gradient; }
                local_dlogits_amax=max_f32(local_dlogits_amax,abs_f32(target_gradient));
            }

            if let Some(row_dlogits_amax) = unsafe {
                block_max_leader_f32(&mut REDUCE, local_dlogits_amax, lane, warp_in_block)
            } {
                unsafe {
                    *dlogits_row_amax.get_unchecked_mut(row as usize) = row_dlogits_amax;
                }
            }
        }
    }
}
