use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

mod score;

use crate::block_reduce::block_max_pair_shared_f32;
use crate::float_ptx::sqrt_f32;
use crate::warp_reduce::thread_lane_warp;
use score::{ClipParams, clip_factor, qk_norms};

use super::threads::{MATRIX_THREADS_PER_BLOCK, WARPS_PER_BLOCK};

#[cuda_module]
pub(super) mod module {
    use super::*;

    #[kernel]
    pub fn kda_muon_qk_clip_kernel(
        qkv: &[u16],
        qk_norm_max: &[f32],
        mut z_master: DisjointSlice<f32>,
        mut x_master: DisjointSlice<f32>,
        mut momentum: DisjointSlice<f32>,
        mut scores: DisjointSlice<f32>,
        row_count: u32,
        qkv_dim: u32,
        input_dim: u32,
        embedding_dim: u32,
        head_count: u32,
        head_dim: u32,
        tau: f32,
        silu_qk: u32,
        norm_offset: u32,
        precomputed_qk_norms: u32,
    ) {
        let head = thread::blockIdx_x();
        if head >= head_count {
            return;
        }

        static mut REDUCE: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;
        static mut REDUCE_PAIR: SharedArray<f32, { WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;
        let (tid, lane, warp_id) = thread_lane_warp();
        let params = ClipParams {
            qkv_dim,
            embedding_dim,
            head_dim,
        };
        let (q_max, k_max) = if precomputed_qk_norms != 0 {
            (
                qk_norm_max[(norm_offset + head) as usize],
                qk_norm_max[(norm_offset + head_count + head) as usize],
            )
        } else {
            let mut q_max = 0.0;
            let mut k_max = 0.0;
            let mut row = tid;
            while row < row_count {
                let (q_norm, k_norm) = qk_norms(qkv, row, head, params, silu_qk);
                q_max = if q_norm > q_max { q_norm } else { q_max };
                k_max = if k_norm > k_max { k_norm } else { k_max };
                row += MATRIX_THREADS_PER_BLOCK;
            }
            unsafe {
                block_max_pair_shared_f32(
                    &mut REDUCE,
                    &mut REDUCE_PAIR,
                    q_max,
                    k_max,
                    lane,
                    warp_id,
                )
            }
        };
        let score = q_max * k_max / sqrt_f32(head_dim as f32);
        let factor = clip_factor(score, tau);
        if tid == 0 {
            unsafe {
                *scores.get_unchecked_mut(head as usize) = score;
            }
        }
        if factor == 1.0 {
            return;
        }

        let per_side = head_dim * input_dim;
        let total = per_side * 2;
        let mut index = tid;
        while index < total {
            let side = index / per_side;
            let rem = index - side * per_side;
            let dim = rem / input_dim;
            let input_row = rem - dim * input_dim;
            let offset = if side == 0 { 0 } else { embedding_dim };
            let col = offset + head * head_dim + dim;
            let master_index = (col * input_dim + input_row) as usize;
            unsafe {
                let z = z_master.get_unchecked_mut(master_index);
                let x = x_master.get_unchecked_mut(master_index);
                let m = momentum.get_unchecked_mut(master_index);
                *z *= factor;
                *x *= factor;
                *m *= factor;
            }
            index += MATRIX_THREADS_PER_BLOCK;
        }
    }
}

pub(super) const KDA_CLIP_THREADS_PER_BLOCK: u32 = MATRIX_THREADS_PER_BLOCK;
