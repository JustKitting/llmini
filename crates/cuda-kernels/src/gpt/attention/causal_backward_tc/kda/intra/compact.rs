use super::super::finish::{
    FinishNormAcc, FinishNormStats, finish_kda_beta_value, finish_kda_qkv_values, kda_norm_index,
};
use super::{
    KDA_INTRA_DG_TILE_ELEMS, KDA_INTRA_DKG_KG_TILE_ELEMS, KDA_INTRA_PAIR_STAT_ELEMS,
    KDA_INTRA_PAIR_VALUE_ELEMS, KDA_INTRA_THREADS_PER_BLOCK, KdaIntraCtx, KdaIntraGrads,
    KdaIntraInputs,
};
use crate::f16_tc_matmul::convert::load_f32_global_read_only;
use crate::float_ptx::{fma_f32, max_f32};
use crate::kda_common::{kda_decay_exp, kda_decay_exp_active, safe_denom};
use crate::kda_elementwise::{KdaQkAct, KdaQkNormAcc, KdaWarpCtx, read_qk_act};
use crate::warp_reduce::warp_sum_f32;
use cuda_device::{SharedArray, thread, warp};

#[derive(Clone, Copy)]
struct CompactGrad {
    dq: f32,
    dk: f32,
    dv: f32,
    db_k: f32,
    db_v: f32,
}

#[derive(Clone, Copy)]
struct CompactPoint {
    dim: u32,
    qk: KdaQkAct,
    grad: CompactGrad,
}

const PAIR_Q_ACT: usize = 0;
const PAIR_K_ACT: usize = 1;
const PAIR_DQ: usize = 2;
const PAIR_DK: usize = 3;
const PAIR_DB_K: usize = 4;
const PAIR_DB_V: usize = 5;
const PAIR_VALUE_FIELDS: usize = 6;

const STAT_Q_NORM: usize = 0;
const STAT_K_NORM: usize = 1;
const STAT_Q_DOT: usize = 2;
const STAT_K_DOT: usize = 3;
const STAT_FIELDS: usize = 4;

#[inline(always)]
fn pair_value_index(token_slot: u32, field: usize, lane: u32) -> usize {
    ((token_slot as usize * PAIR_VALUE_FIELDS + field) * 32) + lane as usize
}

#[inline(always)]
fn pair_stat_index(token_slot: u32, field: usize) -> usize {
    token_slot as usize * STAT_FIELDS + field
}

pub(super) fn update_compact_grads(
    inputs: KdaIntraInputs<'_>,
    grads: KdaIntraGrads,
    dg_tile: &mut SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS>,
    dkg_kg_tile: &mut SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS>,
    pair_values: &mut SharedArray<f32, KDA_INTRA_PAIR_VALUE_ELEMS>,
    pair_stats: &mut SharedArray<f32, KDA_INTRA_PAIR_STAT_ELEMS>,
    ctx: KdaIntraCtx<'_>,
    tid: u32,
    accumulate_value_grad: u32,
) -> f32 {
    let lane = warp::lane_id();
    let warp_in_block = tid / 32;
    let half = warp_in_block & 1;
    let token_slot = warp_in_block >> 1;
    let token_slots = KDA_INTRA_THREADS_PER_BLOCK / 64;
    let dim = lane + half * 32;
    let mut local_amax = 0.0;
    let iterations = 64 / token_slots;
    let mut iteration = 0;
    while iteration < iterations {
        let token_in_chunk = token_slot + iteration * token_slots;
        let token = ctx.start + token_in_chunk;
        let row = ctx.batch * ctx.params.seq_len + token;
        let warp_ctx = KdaWarpCtx {
            lane,
            row,
            batch: ctx.batch,
            token,
            head: ctx.head,
            valid: true,
        };
        let point = update_compact_point(inputs, dg_tile, dkg_kg_tile, ctx, token, dim);

        if half != 0 {
            pair_values[pair_value_index(token_slot, PAIR_Q_ACT, lane)] = point.qk.q_act();
            pair_values[pair_value_index(token_slot, PAIR_K_ACT, lane)] = point.qk.k_act();
            pair_values[pair_value_index(token_slot, PAIR_DQ, lane)] = point.grad.dq;
            pair_values[pair_value_index(token_slot, PAIR_DK, lane)] = point.grad.dk;
            pair_values[pair_value_index(token_slot, PAIR_DB_K, lane)] = point.grad.db_k;
            pair_values[pair_value_index(token_slot, PAIR_DB_V, lane)] = point.grad.db_v;
        }
        thread::sync_threads();

        if half == 0 {
            let warp_ctx = KdaWarpCtx {
                lane,
                row,
                batch: ctx.batch,
                token,
                head: ctx.head,
                valid: true,
            };
            let mut norm_acc = FinishNormAcc::zero();
            norm_acc.add(point.qk, point.grad.dq, point.grad.dk);
            norm_acc.add_values(
                pair_values[pair_value_index(token_slot, PAIR_Q_ACT, lane)],
                pair_values[pair_value_index(token_slot, PAIR_K_ACT, lane)],
                pair_values[pair_value_index(token_slot, PAIR_DQ, lane)],
                pair_values[pair_value_index(token_slot, PAIR_DK, lane)],
            );
            let db_k = point.grad.db_k + pair_values[pair_value_index(token_slot, PAIR_DB_K, lane)];
            let db_v = point.grad.db_v + pair_values[pair_value_index(token_slot, PAIR_DB_V, lane)];
            let db_value = warp_sum_f32(db_k) + warp_sum_f32(db_v);
            let stats = norm_acc.stats();
            if lane == 0 {
                let beta_amax =
                    finish_kda_beta_value(inputs.qkv, grads.d_qkv, warp_ctx, db_value, ctx.params);
                let norm_index = kda_norm_index(warp_ctx, ctx.params);
                unsafe {
                    *grads.q_norms.add(norm_index) = stats.q_norm;
                    *grads.k_norms.add(norm_index) = stats.k_norm;
                }
                local_amax = max_f32(local_amax, beta_amax);
                pair_stats[pair_stat_index(token_slot, STAT_Q_NORM)] = stats.q_norm;
                pair_stats[pair_stat_index(token_slot, STAT_K_NORM)] = stats.k_norm;
                pair_stats[pair_stat_index(token_slot, STAT_Q_DOT)] = stats.q_dot;
                pair_stats[pair_stat_index(token_slot, STAT_K_DOT)] = stats.k_dot;
            }
        }
        thread::sync_threads();

        {
            let stats = FinishNormStats {
                q_norm: pair_stats[pair_stat_index(token_slot, STAT_Q_NORM)],
                k_norm: pair_stats[pair_stat_index(token_slot, STAT_K_NORM)],
                q_dot: pair_stats[pair_stat_index(token_slot, STAT_Q_DOT)],
                k_dot: pair_stats[pair_stat_index(token_slot, STAT_K_DOT)],
            };
            local_amax = max_f32(
                local_amax,
                finish_compact_point(
                    inputs.qkv,
                    grads,
                    warp_ctx,
                    point,
                    stats,
                    accumulate_value_grad,
                    ctx,
                ),
            );
        }
        iteration += 1;
    }
    local_amax
}

#[inline(always)]
fn update_compact_point(
    inputs: KdaIntraInputs<'_>,
    dg_tile: &mut SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS>,
    dkg_kg_tile: &mut SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS>,
    ctx: KdaIntraCtx<'_>,
    token: u32,
    dim: u32,
) -> CompactPoint {
    let row = ctx.batch * ctx.params.seq_len + token;
    let lane = dim & 31;
    let qk0 = read_qk_act(inputs.qkv, row, ctx.head, lane, ctx.params);
    let qk1 = read_qk_act(inputs.qkv, row, ctx.head, lane + 32, ctx.params);
    let mut norm = KdaQkNormAcc::zero();
    norm.add(qk0);
    norm.add(qk1);
    let (_, k_norm) = norm.norms();
    let qk = if dim < 32 { qk0 } else { qk1 };
    // A clamped exponential is not invertible. Recover normalized K from the
    // same raw tape/normalization as forward, never by undoing the Kg factor.
    let k_value = qk.k_act() * (1. / safe_denom(k_norm));
    let grad = update_compact_grad(inputs, dg_tile, dkg_kg_tile, ctx, token, dim, k_value);
    CompactPoint { dim, qk, grad }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn finish_compact_point(
    qkv: &[u16],
    grads: KdaIntraGrads,
    warp_ctx: KdaWarpCtx,
    point: CompactPoint,
    stats: super::super::finish::FinishNormStats,
    accumulate_value_grad: u32,
    ctx: KdaIntraCtx<'_>,
) -> f32 {
    finish_kda_qkv_values(
        qkv,
        grads.d_qkv,
        warp_ctx,
        point.dim,
        point.qk,
        point.grad.dq,
        point.grad.dk,
        point.grad.dv,
        stats,
        accumulate_value_grad,
        ctx.params,
    )
}

fn update_compact_grad(
    inputs: KdaIntraInputs<'_>,
    dg_tile: &mut SharedArray<f32, KDA_INTRA_DG_TILE_ELEMS>,
    dkg_kg_tile: &mut SharedArray<f32, KDA_INTRA_DKG_KG_TILE_ELEMS>,
    ctx: KdaIntraCtx<'_>,
    token: u32,
    dim: u32,
    k_value: f32,
) -> CompactGrad {
    let compact = ctx.compact(token, dim);
    let g_value = load_f32_global_read_only(inputs.g.as_ptr(), compact);
    let g_last = inputs.g_last_chunk[dim as usize];
    let beta_value = inputs.beta_chunk[(token - ctx.start) as usize];
    let exp_g = kda_decay_exp(g_value);
    let exp_neg_g = kda_decay_exp(-g_value);
    let qg_value = load_f32_global_read_only(inputs.qg.as_ptr(), compact);
    let kg_value = load_f32_global_read_only(inputs.kg.as_ptr(), compact);
    let kneg_value = k_value * exp_neg_g;
    let kpos_value = k_value * beta_value * exp_g;

    let d_qg_value = load_f32_global_read_only(inputs.d_qg.as_ptr(), compact);
    let d_k_a_value = load_f32_global_read_only(inputs.d_k_a.as_ptr(), compact);
    let d_kpos_m_value = load_f32_global_read_only(inputs.d_kpos_m.as_ptr(), compact);
    let tile_index = ((token - ctx.start) * 64 + dim) as usize;
    // Consume both projection cells before either same cell changes roles.
    // Read through the mutable arenas themselves, without aliased references.
    let d_kneg_b_value = dkg_kg_tile[tile_index];
    let d_kpos_b_t_value = dg_tile[tile_index];
    // The detached API previously filled the complete global dKG tensor with
    // +0.0. Supply those exact bits here without a store/read round trip.
    let d_kg_value = if inputs.detached_chunk_state != 0 {
        0.0
    } else {
        load_f32_global_read_only(inputs.d_kg.as_ptr(), compact)
    };
    let state_g = if kda_decay_exp_active(g_last - g_value) {
        d_kg_value * kg_value
    } else {
        0.
    };
    dkg_kg_tile[tile_index] = state_g;
    let mut dk_value = d_kg_value * kda_decay_exp(g_last - g_value)
        + d_k_a_value * exp_neg_g
        + d_kpos_m_value * beta_value * exp_g;
    let d_vbeta_m_value = load_f32_global_read_only(inputs.d_vbeta_m.as_ptr(), compact);
    let dv_value = d_vbeta_m_value * beta_value;
    let mut dg_value = -d_k_a_value * kneg_value + d_kpos_m_value * kpos_value;

    dk_value = fma_f32(d_kpos_b_t_value, exp_neg_g, dk_value);
    dk_value = fma_f32(d_kneg_b_value, beta_value * exp_g, dk_value);
    dg_value -= d_kpos_b_t_value * kneg_value;
    dg_value = fma_f32(d_kneg_b_value, kpos_value, dg_value);

    dg_tile[tile_index] = if kda_decay_exp_active(g_value) {
        dg_value + d_qg_value * qg_value
    } else {
        0.
    } - state_g;

    let d_kpos = d_kpos_m_value + d_kneg_b_value;
    let v_value =
        load_f32_global_read_only(inputs.vbeta.as_ptr(), compact) / safe_denom(beta_value);
    CompactGrad {
        dq: d_qg_value * exp_g,
        dk: dk_value,
        dv: dv_value,
        db_k: d_kpos * k_value * exp_g,
        db_v: d_vbeta_m_value * v_value,
    }
}
