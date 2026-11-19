use cuda_device::{
    DisjointSlice, SharedArray,
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, kernel, thread,
};

use crate::block_reduce::block_max_leader_f32;
use crate::float_ptx::{abs_f32, sqrt_f32};
use crate::warp_reduce::thread_lane_warp;

use super::threads::{APPLY_THREADS_PER_BLOCK, WARPS_PER_BLOCK};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn zero_f32_pointer_table_kernel(pointers: &[u64], pointer_count: u32) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < pointer_count {
            unsafe {
                *(pointers[index as usize] as usize as *mut f32) = 0.0;
            }
        }
    }

    #[kernel]
    pub fn fp32_adamw_update_kernel(
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        grad: &[f32],
        grad_scale: f32,
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
        len: u32,
    ) {
        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        if index < len {
            let i = index as usize;
            let g = grad[i] * grad_scale;

            unsafe {
                let z_master = z_master.as_mut_ptr().add(i);
                let x_master = x_master.as_mut_ptr().add(i);
                let first = first_moment.as_mut_ptr().add(i);
                let second = second_moment.as_mut_ptr().add(i);
                let m = beta1 * *first + (1.0 - beta1) * g;
                let v = beta2 * *second + (1.0 - beta2) * g * g;
                let update = (m / beta1_correction) / (sqrt_f32(v / beta2_correction) + eps);
                let current = *z_master;
                let decay = 1.0 - learning_rate * weight_decay;
                let next = current * decay - learning_rate * update;
                let x = *x_master;

                *first = m;
                *second = v;
                *z_master = next;
                *x_master = x + average_coefficient * (next - x);
            }
        }
    }

    #[kernel]
    pub fn fp32_adamw_update_schedule_amax_kernel(
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        mut grad: DisjointSlice<f32>,
        grad_scale: f32,
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        output_amax: &[f32],
        learning_rate: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        beta1_correction: f32,
        beta2_correction: f32,
        eps: f32,
        average_coefficient: f32,
        schedule_beta: f32,
        len: u32,
    ) {
        static mut CTA_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

        let index = thread::blockIdx_x() * APPLY_THREADS_PER_BLOCK + thread::threadIdx_x();
        let mut local_amax = 0.0;
        if index < len {
            let i = index as usize;
            let g = unsafe { *grad.as_mut_ptr().add(i) } * grad_scale;

            unsafe {
                let z_master = z_master.as_mut_ptr().add(i);
                let x_master = x_master.as_mut_ptr().add(i);
                let first = first_moment.as_mut_ptr().add(i);
                let second = second_moment.as_mut_ptr().add(i);
                let m = beta1 * *first + (1.0 - beta1) * g;
                let v = beta2 * *second + (1.0 - beta2) * g * g;
                let update = (m / beta1_correction) / (sqrt_f32(v / beta2_correction) + eps);
                let current = *z_master;
                let decay = 1.0 - learning_rate * weight_decay;
                let next = current * decay - learning_rate * update;
                let x = *x_master;
                let next_x = x + average_coefficient * (next - x);

                *first = m;
                *second = v;
                *z_master = next;
                *x_master = next_x;

                let scheduled = next + schedule_beta * (next_x - next);
                *grad.as_mut_ptr().add(i) = scheduled;
                local_amax = abs_f32(scheduled);
            }
        }

        let (_, lane, warp_in_block) = thread_lane_warp();
        if let Some(cta_amax) =
            unsafe { block_max_leader_f32(&mut CTA_AMAX, local_amax, lane, warp_in_block) }
        {
            let output = unsafe { &*(output_amax.as_ptr() as *const DeviceAtomicU32) };
            output.fetch_max(cta_amax.to_bits(), AtomicOrdering::Relaxed);
        }
    }
}
