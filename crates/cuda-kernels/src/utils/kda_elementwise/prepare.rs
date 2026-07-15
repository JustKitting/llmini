use cuda_device::{DisjointSlice, thread};

use crate::attention::CausalAttentionParams;
use crate::f16_tc_matmul::convert::cvt_rn_f16_f32;
use crate::kda_common::{
    batch_head, beta_compact_index, beta_index, beta_offset, compact_index, g_offset, k_offset,
    q_offset, qkv_index, safe_denom, sigmoid, silu, softplus, v_offset,
};

use super::context::{KdaQkAct, KdaQkNormAcc, KdaQkvRead, KdaWarpCtx, kda_warp_ctx, read_qk_act};

pub(crate) struct KdaPrepareOutputs<'a> {
    pub(crate) q: DisjointSlice<'a, f32>,
    pub(crate) k: DisjointSlice<'a, f32>,
    pub(crate) v: DisjointSlice<'a, f32>,
    pub(crate) g: DisjointSlice<'a, f32>,
    pub(crate) beta: DisjointSlice<'a, f32>,
}

pub(crate) fn chunk_cumsum_g_body(mut g: DisjointSlice<f32>, params: CausalAttentionParams) {
    let bh = thread::blockIdx_x();
    let chunk = thread::blockIdx_y();
    let dim = thread::threadIdx_x();
    if bh >= batch_head(&params)
        || chunk * params.chunk_size >= params.seq_len
        || dim >= params.head_dim
    {
        return;
    }

    let batch = bh / params.head_count;
    let head = bh - batch * params.head_count;
    let start = chunk * params.chunk_size;
    let end = params.seq_len.min(start + params.chunk_size);
    let mut acc = 0.0;
    let mut token = start;
    while token < end {
        let compact = compact_index(batch, token, head, dim, &params);
        unsafe {
            acc += *g.get_unchecked_mut(compact);
            *g.get_unchecked_mut(compact) = acc;
        }
        token += 1;
    }
}

pub(crate) fn prepare_kda_inputs_body<T: KdaQkvRead>(
    qkv: &[T],
    mut out: KdaPrepareOutputs<'_>,
    qkv_f16: *mut u16,
    params: CausalAttentionParams,
    threads_per_block: u32,
) {
    let ctx = kda_warp_ctx(threads_per_block, &params);
    if !ctx.valid {
        return;
    }

    let mut acc = KdaQkNormAcc::zero();
    let dim0 = ctx.lane;
    let qk0 = read_prepare_dim(qkv, ctx, dim0, &params, &mut acc);
    let dim1 = ctx.lane + 32;
    let qk1 = read_prepare_dim(qkv, ctx, dim1, &params, &mut acc);
    let (q_norm, k_norm) = acc.norms();
    let inv = (params.scale / safe_denom(q_norm), 1.0 / safe_denom(k_norm));

    if dim0 < params.head_dim {
        write_prepared(qkv, &mut out, qkv_f16, ctx, dim0, qk0, inv, &params);
    }
    if dim1 < params.head_dim {
        write_prepared(qkv, &mut out, qkv_f16, ctx, dim1, qk1, inv, &params);
    }
    if ctx.lane == 0 {
        let beta_index = beta_index(ctx.row, ctx.head, &params);
        let raw_beta = T::read(qkv, beta_index);
        unsafe {
            *out.beta
                .get_unchecked_mut(beta_compact_index(ctx.batch, ctx.token, ctx.head, &params)) =
                sigmoid(raw_beta);
        }
        store_qkv_f16(qkv_f16, beta_index, raw_beta);
    }

    // The aligned KDA projection has a short inactive tail after beta. Spread
    // that exact tape copy across the row's head warps so the fused path still
    // preserves every padded projection value, not just the values KDA reads.
    let padding_index = ctx.head * 32 + ctx.lane;
    let active_qkv = beta_offset(&params) + params.head_count;
    if active_qkv + padding_index < params.qkv_dim {
        let index = (ctx.row * params.qkv_dim + active_qkv + padding_index) as usize;
        store_qkv_f16(qkv_f16, index, T::read(qkv, index));
    }
}

#[inline(always)]
fn read_prepare_dim<T: KdaQkvRead>(
    qkv: &[T],
    ctx: KdaWarpCtx,
    dim: u32,
    params: &CausalAttentionParams,
    acc: &mut KdaQkNormAcc,
) -> KdaQkAct {
    if dim >= params.head_dim {
        return KdaQkAct::zero();
    }

    let qk = read_qk_act(qkv, ctx.row, ctx.head, dim, params);
    acc.add(qk);
    qk
}

fn write_prepared<T: KdaQkvRead>(
    qkv: &[T],
    out: &mut KdaPrepareOutputs<'_>,
    qkv_f16: *mut u16,
    ctx: KdaWarpCtx,
    dim: u32,
    qk: KdaQkAct,
    inv: (f32, f32),
    params: &CausalAttentionParams,
) {
    let (q_inv, k_inv) = inv;
    let compact = compact_index(ctx.batch, ctx.token, ctx.head, dim, params);
    let q_index = qkv_index(ctx.row, ctx.head, dim, q_offset(params), params);
    let k_index = qkv_index(ctx.row, ctx.head, dim, k_offset(params), params);
    let v_index = qkv_index(ctx.row, ctx.head, dim, v_offset(params), params);
    let g_index = qkv_index(ctx.row, ctx.head, dim, g_offset(params), params);
    let raw_v = T::read(qkv, v_index);
    let raw_g = T::read(qkv, g_index);
    unsafe {
        *out.q.get_unchecked_mut(compact) = qk.q_act * q_inv;
        *out.k.get_unchecked_mut(compact) = qk.k_act * k_inv;
        *out.v.get_unchecked_mut(compact) = silu(raw_v);
        *out.g.get_unchecked_mut(compact) = -params.decay_scale * softplus(raw_g);
    }
    store_qkv_f16(qkv_f16, q_index, qk.raw_q);
    store_qkv_f16(qkv_f16, k_index, qk.raw_k);
    store_qkv_f16(qkv_f16, v_index, raw_v);
    store_qkv_f16(qkv_f16, g_index, raw_g);
}

#[inline(always)]
fn store_qkv_f16(values: *mut u16, index: usize, value: f32) {
    if !values.is_null() {
        unsafe {
            *values.add(index) = cvt_rn_f16_f32(value);
        }
    }
}
