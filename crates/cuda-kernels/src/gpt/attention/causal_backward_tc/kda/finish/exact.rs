//! Original FP32 affine-input specialization of the reached shared normalizer,
//! SiLU and gate VJPs. Keep value accumulation and final core amax ownership.
use super::{FinishNormAcc, finish_kda_beta_value, finish_kda_g_value, finish_kda_qkv_values};
use crate::attention::CausalAttentionParams;
use crate::block_reduce::block_max_leader_f32;
use crate::float_ptx::{fma_f32, max_f32};
use crate::kda_common::{beta_index, qkv_index, sigmoid, v_offset, value_activation};
use crate::kda_elementwise::{KdaWarpCtx, read_qk_act};
use crate::warp_reduce::warp_sum_f32;
use cuda_device::{DisjointSlice, SharedArray, thread, warp};

pub(crate) fn exact_kda_raw_finish_body(
    qkv: &[f32],
    dq: &[f32],
    dk: &[f32],
    dg: &[f32],
    dvbeta: &[f32],
    dbeta: &[f32],
    mut out: DisjointSlice<f32>,
    mut amax: DisjointSlice<f32>,
    accumulate_value_grad: u32,
    p: CausalAttentionParams,
    amax_shared: &mut SharedArray<f32, 4>,
) {
    let bh = thread::blockIdx_x();
    let chunk = thread::blockIdx_y();
    let chunks = p.seq_len.div_ceil(64);
    let batch = bh / p.head_count;
    let head = bh % p.head_count;
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_id = tid / 32;
    let valid = (p.seq_len - chunk * 64).min(64);
    let mut token = warp_id;
    let mut local_amax = 0.0;
    while token < valid {
        let position = chunk * 64 + token;
        let row = batch * p.seq_len + position;
        let ctx = KdaWarpCtx {
            lane,
            row,
            batch,
            token: position,
            head,
            valid: true,
        };
        let at = (((bh * chunks + chunk) * 64 + token) * 64 + lane) as usize;
        let qk0 = read_qk_act(qkv, row, head, lane, &p);
        let qk1 = read_qk_act(qkv, row, head, lane + 32, &p);
        let mut norm = FinishNormAcc::zero();
        norm.add(qk0, dq[at], dk[at]);
        norm.add(qk1, dq[at + 32], dk[at + 32]);
        let stats = norm.stats();
        let beta = warp::shuffle_f32(
            if lane == 0 {
                sigmoid(qkv[beta_index(row, head, &p)])
            } else {
                0.0
            },
            0,
        );
        local_amax = max_f32(
            local_amax,
            finish_kda_qkv_values(
                qkv,
                out.as_mut_ptr(),
                ctx,
                lane,
                qk0,
                dq[at],
                dk[at],
                beta * dvbeta[at],
                stats,
                accumulate_value_grad,
                &p,
            ),
        );
        local_amax = max_f32(
            local_amax,
            finish_kda_qkv_values(
                qkv,
                out.as_mut_ptr(),
                ctx,
                lane + 32,
                qk1,
                dq[at + 32],
                dk[at + 32],
                beta * dvbeta[at + 32],
                stats,
                accumulate_value_grad,
                &p,
            ),
        );
        local_amax = max_f32(
            local_amax,
            finish_kda_g_value(qkv, out.as_mut_ptr(), ctx, lane, dg[at], &p),
        );
        local_amax = max_f32(
            local_amax,
            finish_kda_g_value(qkv, out.as_mut_ptr(), ctx, lane + 32, dg[at + 32], &p),
        );
        // Beta is outside the value activation/normalization in forward.
        let v0 = value_activation(qkv[qkv_index(row, head, lane, v_offset(&p), &p)]);
        let v1 = value_activation(qkv[qkv_index(row, head, lane + 32, v_offset(&p), &p)]);
        let extra_beta = warp_sum_f32(fma_f32(v0, dvbeta[at], v1 * dvbeta[at + 32]));
        if lane == 0 {
            local_amax = max_f32(
                local_amax,
                finish_kda_beta_value(qkv, out.as_mut_ptr(), ctx, dbeta[at / 64] + extra_beta, &p),
            );
        }
        token += 4;
    }
    let maximum = block_max_leader_f32(amax_shared, local_amax, lane, warp_id);
    if tid < 8 {
        let chunk_index = bh * chunks + chunk;
        let value = if tid == 0 {
            maximum.unwrap_or(0.0)
        } else {
            0.0
        };
        unsafe {
            *amax.get_unchecked_mut((chunk_index * 8 + tid) as usize) = value;
        }
    }
}
