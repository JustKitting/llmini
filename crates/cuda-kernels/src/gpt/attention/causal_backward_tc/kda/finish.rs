use cuda_device::DisjointSlice;

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::amax::max4_f32;
use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_f32_f16;
use crate::float_ptx::{abs_f32, fma_f32, max_f32};
use crate::kda_common::{
    beta_compact_index, beta_index, compact_index, g_offset, k_offset, q_offset, qkv_index,
    safe_denom, sigmoid, silu_grad, v_offset,
};
use crate::kda_elementwise::{KdaQkAct, KdaQkNormAcc, KdaWarpCtx, kda_warp_ctx, read_qk_act};
use crate::warp_reduce::warp_sum_f32;

#[derive(Clone, Copy)]
pub(crate) struct FinishKdaGrads<'a> {
    pub(crate) q: &'a [f32],
    pub(crate) k: &'a [f32],
    pub(crate) v: &'a [f32],
    pub(crate) g: &'a [f32],
    pub(crate) beta: &'a [f32],
}

#[derive(Clone, Copy)]
struct FinishDimPoint {
    ctx: KdaWarpCtx,
    dim: u32,
    qk: KdaQkAct,
}

#[derive(Clone, Copy)]
struct FinishNormStats {
    q_norm: f32,
    k_norm: f32,
    q_dot: f32,
    k_dot: f32,
}

#[derive(Clone, Copy)]
struct FinishNormAcc {
    qk_norm: KdaQkNormAcc,
    q_dot: f32,
    k_dot: f32,
}

impl FinishNormAcc {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            qk_norm: KdaQkNormAcc::zero(),
            q_dot: 0.0,
            k_dot: 0.0,
        }
    }

    #[inline(always)]
    fn stats(self) -> FinishNormStats {
        let (q_norm, k_norm) = self.qk_norm.norms();
        FinishNormStats {
            q_norm,
            k_norm,
            q_dot: warp_sum_f32(self.q_dot),
            k_dot: warp_sum_f32(self.k_dot),
        }
    }
}

pub(crate) fn finish_kda_backward_body(
    qkv: &[u16],
    grads: FinishKdaGrads<'_>,
    mut q_norms: DisjointSlice<f32>,
    mut k_norms: DisjointSlice<f32>,
    mut d_qkv: DisjointSlice<f32>,
    accumulate_value_grad: u32,
    params: CausalAttentionParams,
) -> f32 {
    let ctx = kda_warp_ctx(TC_BACKWARD_THREADS_PER_BLOCK, &params);
    if !ctx.valid {
        return 0.0;
    }

    let mut acc = FinishNormAcc::zero();
    let dim0 = ctx.lane;
    let qk0 = read_finish_dim(qkv, grads, ctx, dim0, &params, &mut acc);
    let dim1 = ctx.lane + 32;
    let qk1 = read_finish_dim(qkv, grads, ctx, dim1, &params, &mut acc);
    let stats = acc.stats();

    let mut local_amax = 0.0;
    if dim0 < params.head_dim {
        local_amax = finish_dim(
            qkv,
            grads,
            &mut d_qkv,
            FinishDimPoint {
                ctx,
                dim: dim0,
                qk: qk0,
            },
            stats,
            accumulate_value_grad,
            &params,
        );
    }
    if dim1 < params.head_dim {
        local_amax = max_f32(
            local_amax,
            finish_dim(
                qkv,
                grads,
                &mut d_qkv,
                FinishDimPoint {
                    ctx,
                    dim: dim1,
                    qk: qk1,
                },
                stats,
                accumulate_value_grad,
                &params,
            ),
        );
    }
    if ctx.lane == 0 {
        let norm_index = (ctx.head * params.row_count + ctx.row) as usize;
        let raw_beta = cvt_f32_f16(qkv[beta_index(ctx.row, ctx.head, &params)]);
        let beta_value = sigmoid(raw_beta);
        let grad = grads.beta[beta_compact_index(ctx.batch, ctx.token, ctx.head, &params)]
            * beta_value
            * (1.0 - beta_value);
        unsafe {
            *q_norms.get_unchecked_mut(norm_index) = stats.q_norm;
            *k_norms.get_unchecked_mut(norm_index) = stats.k_norm;
            *d_qkv.get_unchecked_mut(beta_index(ctx.row, ctx.head, &params)) = grad;
        }
        local_amax = max_f32(local_amax, abs_f32(grad));
    }
    local_amax
}

#[inline(always)]
fn read_finish_dim(
    qkv: &[u16],
    grads: FinishKdaGrads<'_>,
    ctx: KdaWarpCtx,
    dim: u32,
    params: &CausalAttentionParams,
    acc: &mut FinishNormAcc,
) -> KdaQkAct {
    if dim >= params.head_dim {
        return KdaQkAct::zero();
    }

    let qk = read_qk_act(qkv, ctx.row, ctx.head, dim, params);
    let compact = compact_index(ctx.batch, ctx.token, ctx.head, dim, params);
    acc.qk_norm.add(qk);
    acc.q_dot = fma_f32(grads.q[compact], qk.q_act, acc.q_dot);
    acc.k_dot = fma_f32(grads.k[compact], qk.k_act, acc.k_dot);
    qk
}

fn finish_dim(
    qkv: &[u16],
    grads: FinishKdaGrads<'_>,
    d_qkv: &mut DisjointSlice<f32>,
    point: FinishDimPoint,
    stats: FinishNormStats,
    accumulate_value_grad: u32,
    params: &CausalAttentionParams,
) -> f32 {
    let FinishDimPoint { ctx, dim, qk } = point;
    let compact = compact_index(ctx.batch, ctx.token, ctx.head, dim, params);
    let q_denom = safe_denom(stats.q_norm);
    let k_denom = safe_denom(stats.k_norm);
    let q_cubic_denom = safe_denom(stats.q_norm * stats.q_norm * stats.q_norm);
    let k_cubic_denom = safe_denom(stats.k_norm * stats.k_norm * stats.k_norm);
    let dq_norm =
        params.scale * (grads.q[compact] / q_denom - qk.q_act * stats.q_dot / q_cubic_denom);
    let dk_norm = grads.k[compact] / k_denom - qk.k_act * stats.k_dot / k_cubic_denom;
    let raw_v = cvt_f32_f16(qkv[qkv_index(ctx.row, ctx.head, dim, v_offset(params), params)]);
    let raw_g = cvt_f32_f16(qkv[qkv_index(ctx.row, ctx.head, dim, g_offset(params), params)]);
    let dq = dq_norm * silu_grad(qk.raw_q);
    let dk = dk_norm * silu_grad(qk.raw_k);
    let value_index = qkv_index(ctx.row, ctx.head, dim, v_offset(params), params);
    let direct_value_grad = if accumulate_value_grad != 0 {
        unsafe { *d_qkv.as_mut_ptr().add(value_index) }
    } else {
        0.0
    };
    let dv = grads.v[compact] * silu_grad(raw_v) + direct_value_grad;
    let dg = -params.decay_scale * grads.g[compact] * sigmoid(raw_g);
    unsafe {
        *d_qkv.get_unchecked_mut(qkv_index(ctx.row, ctx.head, dim, q_offset(params), params)) = dq;
        *d_qkv.get_unchecked_mut(qkv_index(ctx.row, ctx.head, dim, k_offset(params), params)) = dk;
        *d_qkv.get_unchecked_mut(value_index) = dv;
        *d_qkv.get_unchecked_mut(qkv_index(ctx.row, ctx.head, dim, g_offset(params), params)) = dg;
    }
    max4_f32(abs_f32(dq), abs_f32(dk), abs_f32(dv), abs_f32(dg))
}
