//! KDA adjoint repairs: native RHS query accumulation and direct state decay.
//! Capture state decay before the exported dH arena is reused.
use super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use super::kda::stage_chain_native;
use crate::attention::CausalAttentionParams;
use crate::block_reduce::block_max_leader_f32;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{abs_f32, max_f32};
use crate::kda_common::{KDA_DECAY_EXP_LIMIT, kda_decay_exp, sigmoid};
use crate::warp_reduce::warp_sum_f32;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread, warp};
const WARPS: usize = (TC_BACKWARD_THREADS_PER_BLOCK / 32) as usize;
pub(crate) use module::{LoadedModule, from_module};

#[cuda_module]
pub mod module {
    use super::*;
    /// dQg += dAQK * Kneg. Kneg is token-by-feature, not B-transposed.
    /// Use its original FP16 cells and the established FP32 MMA accumulator.
    #[kernel]
    pub fn accumulate_kda_dqg_rhs_kernel(
        daqk: &[u16],
        kneg: &[u16],
        mut dqg: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut A: crate::kda_tc::CtaATile = SharedArray::UNINIT;
        static mut B: crate::kda_tc::CtaBTile = SharedArray::UNINIT;
        let a = unsafe { &mut A };
        let b = unsafe { &mut B };
        let bh = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let base = ((bh * (params.seq_len / 64) + chunk) * 4096) as usize;
        let tile =
            crate::f16_tc_matmul::cta_tile::CtaTile::from_wide_tile(thread::threadIdx_x(), 0, 0, 0);
        let mut lo = [0f32; 4];
        let mut hi = [0f32; 4];
        let mut k = 0;
        while k < 64 {
            stage_chain_native::<false>(daqk, a, base, k);
            stage_chain_native::<true>(kneg, b, base, k);
            thread::sync_threads();
            crate::kda_tc::mma_accumulate_pair(tile, a, b, &mut lo, &mut hi);
            thread::sync_threads();
            k += 16;
        }
        for i in 0..4 {
            let (row, col) = tile.accumulator_coords(tile.warp_n0, i);
            let (row_hi, col_hi) = tile.accumulator_coords(tile.warp_n0 + 1, i);
            unsafe {
                *dqg.get_unchecked_mut(base + (row * 64 + col) as usize) += lo[i];
                *dqg.get_unchecked_mut(base + (row_hi * 64 + col_hi) as usize) += hi[i];
            }
        }
    }

    #[kernel]
    pub fn capture_full_state_decay_grad_kernel(
        entry: &[u16],
        dh_next: &[u16],
        g: &[f32],
        mut state_grad: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        let bh = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let chunks = params.seq_len / 64;
        let lane = warp::lane_id();
        let warp_in_block = thread::threadIdx_x() / 32;
        let base = ((bh * chunks + chunk) * 4096) as usize;
        let mut key = warp_in_block;
        while key < 64 {
            let i = base + (key * 64 + lane) as usize;
            let sum = cvt_f32_f16(entry[i]) * cvt_f32_f16(dh_next[i])
                + cvt_f32_f16(entry[i + 32]) * cvt_f32_f16(dh_next[i + 32]);
            let dot = warp_sum_f32(sum);
            if lane == 0 {
                let last = g[((bh * params.seq_len + chunk * 64 + 63) * 64 + key) as usize];
                let value = if last < -KDA_DECAY_EXP_LIMIT || last > KDA_DECAY_EXP_LIMIT {
                    0.
                } else {
                    kda_decay_exp(last) * dot
                };
                unsafe {
                    *state_grad.get_unchecked_mut(((bh * chunks + chunk) * 64 + key) as usize) =
                        value;
                }
            }
            key += thread::blockDim_x() / 32;
        }
    }
    #[kernel]
    pub fn add_full_state_decay_grad_and_amax_kernel(
        qkv: &[u16],
        state_grad: &[f32],
        mut dqkv: DisjointSlice<f32>,
        mut amax: DisjointSlice<f32>,
        params: CausalAttentionParams,
    ) {
        static mut MAX: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        let bh = thread::blockIdx_x();
        let chunk = thread::blockIdx_y();
        let chunks = params.seq_len / 64;
        let batch = bh / params.head_count;
        let head = bh % params.head_count;
        let tid = thread::threadIdx_x();
        let section = tid / 64;
        let dim = tid % 64;
        let mut local = 0f32;
        for token in 0..64 {
            let row = batch * params.seq_len + chunk * 64 + token;
            let i =
                (row * params.qkv_dim + section * params.embedding_dim + head * 64 + dim) as usize;
            let mut grad = unsafe { *dqkv.get_unchecked_mut(i) };
            if section == 3 {
                let extra = state_grad[((bh * chunks + chunk) * 64 + dim) as usize];
                grad += -params.decay_scale * extra * sigmoid(cvt_f32_f16(qkv[i]));
                unsafe {
                    *dqkv.get_unchecked_mut(i) = grad;
                }
            }
            local = max_f32(local, abs_f32(grad));
            if tid == 0 {
                let beta = (row * params.qkv_dim + 4 * params.embedding_dim + head) as usize;
                local = max_f32(local, abs_f32(unsafe { *dqkv.get_unchecked_mut(beta) }));
            }
        }
        let maximum = unsafe { block_max_leader_f32(&mut MAX, local, warp::lane_id(), tid / 32) };
        if tid < WARPS as u32 {
            unsafe {
                *amax.get_unchecked_mut(((bh * chunks + chunk) * WARPS as u32 + tid) as usize) =
                    if tid == 0 { maximum.unwrap_or(0.) } else { 0. };
            }
        }
    }
}
