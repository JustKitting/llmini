use cuda_device::DisjointSlice;
mod exact;
pub(crate) use exact::exact_kda_raw_finish_body;

use super::super::gather::TC_BACKWARD_THREADS_PER_BLOCK;
use crate::amax::max4_f32;
use crate::attention::CausalAttentionParams;
use crate::float_ptx::{abs_f32, fma_f32, max_f32};
use crate::kda_common::{
    beta_compact_index, beta_index, compact_index, g_offset, k_offset, q_offset, qkv_index,
    safe_denom, sigmoid, v_offset, value_activation_grad,
};
use crate::kda_elementwise::{
    KdaQkAct, KdaQkNormAcc, KdaQkvRead, KdaWarpCtx, kda_warp_ctx, read_qk_act,
};
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
pub(crate) struct FinishNormStats {
    pub(crate) q_norm: f32,
    pub(crate) k_norm: f32,
    pub(crate) q_dot: f32,
    pub(crate) k_dot: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct FinishNormAcc {
    qk_norm: KdaQkNormAcc,
    q_dot: f32,
    k_dot: f32,
}

impl FinishNormAcc {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            qk_norm: KdaQkNormAcc::zero(),
            q_dot: 0.0,
            k_dot: 0.0,
        }
    }

    #[inline(always)]
    pub(crate) fn add(&mut self, qk: KdaQkAct, dq: f32, dk: f32) {
        self.add_values(qk.q_act(), qk.k_act(), dq, dk);
    }

    #[inline(always)]
    pub(crate) fn add_values(&mut self, q_act: f32, k_act: f32, dq: f32, dk: f32) {
        self.qk_norm.add_values(q_act, k_act);
        self.q_dot = fma_f32(dq, q_act, self.q_dot);
        self.k_dot = fma_f32(dk, k_act, self.k_dot);
    }

    #[inline(always)]
    pub(crate) fn stats(self) -> FinishNormStats {
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
        let norm_index = kda_norm_index(ctx, &params);
        let grad = finish_kda_beta_value(
            qkv,
            d_qkv.as_mut_ptr(),
            ctx,
            grads.beta[beta_compact_index(ctx.batch, ctx.token, ctx.head, &params)],
            &params,
        );
        unsafe {
            *q_norms.get_unchecked_mut(norm_index) = stats.q_norm;
            *k_norms.get_unchecked_mut(norm_index) = stats.k_norm;
        }
        local_amax = max_f32(local_amax, grad);
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
    acc.add(qk, grads.q[compact], grads.k[compact]);
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
    let mut local_amax = finish_kda_qkv_values(
        qkv,
        d_qkv.as_mut_ptr(),
        ctx,
        dim,
        qk,
        grads.q[compact],
        grads.k[compact],
        grads.v[compact],
        stats,
        accumulate_value_grad,
        params,
    );
    local_amax = max_f32(
        local_amax,
        finish_kda_g_value(qkv, d_qkv.as_mut_ptr(), ctx, dim, grads.g[compact], params),
    );
    local_amax
}

#[inline(always)]
pub(crate) fn kda_norm_index(ctx: KdaWarpCtx, params: &CausalAttentionParams) -> usize {
    (ctx.head * params.row_count + ctx.row) as usize
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_kda_qkv_values<T: KdaQkvRead>(
    qkv: &[T],
    d_qkv: *mut f32,
    ctx: KdaWarpCtx,
    dim: u32,
    qk: KdaQkAct,
    dq_grad: f32,
    dk_grad: f32,
    dv_grad: f32,
    stats: FinishNormStats,
    accumulate_value_grad: u32,
    params: &CausalAttentionParams,
) -> f32 {
    let q_denom = safe_denom(stats.q_norm);
    let k_denom = safe_denom(stats.k_norm);
    // Forward uses n=sqrt(sum(a*a)+eps), then a/(n+eps). The radial
    // denominator is n*(n+eps)^2. Adding eps to n^3 instead changes that
    // derivative drastically for small norms; n is already >=sqrt(eps).
    let q_radial_denom = stats.q_norm * q_denom * q_denom;
    let k_radial_denom = stats.k_norm * k_denom * k_denom;
    let dq_norm = params.scale * (dq_grad / q_denom - qk.q_act() * stats.q_dot / q_radial_denom);
    let dk_norm = dk_grad / k_denom - qk.k_act() * stats.k_dot / k_radial_denom;
    let raw_v = T::read(
        qkv,
        qkv_index(ctx.row, ctx.head, dim, v_offset(params), params),
    );
    let dq = dq_norm * qk.q_silu_grad();
    let dk = dk_norm * qk.k_silu_grad();
    let value_index = qkv_index(ctx.row, ctx.head, dim, v_offset(params), params);
    let direct_value_grad = if accumulate_value_grad != 0 {
        unsafe { *d_qkv.add(value_index) }
    } else {
        0.0
    };
    let dv = dv_grad * value_activation_grad(raw_v) + direct_value_grad;
    unsafe {
        *d_qkv.add(qkv_index(ctx.row, ctx.head, dim, q_offset(params), params)) = dq;
        *d_qkv.add(qkv_index(ctx.row, ctx.head, dim, k_offset(params), params)) = dk;
        *d_qkv.add(value_index) = dv;
    }
    max4_f32(abs_f32(dq), abs_f32(dk), abs_f32(dv), 0.0)
}

#[inline(always)]
pub(crate) fn finish_kda_g_value<T: KdaQkvRead>(
    qkv: &[T],
    d_qkv: *mut f32,
    ctx: KdaWarpCtx,
    dim: u32,
    dg_grad: f32,
    params: &CausalAttentionParams,
) -> f32 {
    let index = qkv_index(ctx.row, ctx.head, dim, g_offset(params), params);
    let raw_g = T::read(qkv, index);
    let dg = -params.decay_scale * dg_grad * sigmoid(raw_g);
    unsafe {
        *d_qkv.add(index) = dg;
    }
    abs_f32(dg)
}

#[inline(always)]
pub(crate) fn finish_kda_beta_value<T: KdaQkvRead>(
    qkv: &[T],
    d_qkv: *mut f32,
    ctx: KdaWarpCtx,
    db_grad: f32,
    params: &CausalAttentionParams,
) -> f32 {
    let index = beta_index(ctx.row, ctx.head, params);
    let raw_beta = T::read(qkv, index);
    let beta_value = sigmoid(raw_beta);
    let grad = db_grad * beta_value * (1.0 - beta_value);
    unsafe {
        *d_qkv.add(index) = grad;
    }
    abs_f32(grad)
}
