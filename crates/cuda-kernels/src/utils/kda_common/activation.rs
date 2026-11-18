use crate::attention::CausalAttentionParams;
use crate::float_ptx::{exp_f32, ln_f32, sqrt_f32};
use crate::warp_reduce::warp_sum_f32;

use super::shape::KDA_DENOM_EPS;

pub(crate) const KDA_DECAY_EXP_LIMIT: f32 = 3.0;

// The separate value-RMS graph already applied SiLU before head normalization.
// Beta remains outside that normalization in every preparation/replay path.
#[inline(always)]
pub(crate) fn value_activation(x: f32) -> f32 {
    if crate::value_rms::config::VALUE_RMS {
        x
    } else {
        silu(x)
    }
}
#[inline(always)]
pub(crate) fn value_activation_grad(x: f32) -> f32 {
    if crate::value_rms::config::VALUE_RMS {
        1.0
    } else {
        silu_grad(x)
    }
}

#[inline(always)]
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline(always)]
pub(crate) fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline(always)]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}

#[inline(always)]
pub(crate) fn safe_denom(x: f32) -> f32 {
    if x >= 0.0 {
        x + KDA_DENOM_EPS
    } else {
        x - KDA_DENOM_EPS
    }
}

#[inline(always)]
pub(crate) fn kda_warp_norm(sum: f32) -> f32 {
    sqrt_f32(warp_sum_f32(sum) + KDA_DENOM_EPS)
}

#[inline(always)]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        ln_f32(1.0 + exp_f32(x))
    }
}

#[inline(always)]
#[expect(clippy::manual_clamp, reason = "cuda-oxide does not lower f32::clamp")]
pub(crate) fn kda_decay_exp(x: f32) -> f32 {
    let x = match x {
        x if x < -KDA_DECAY_EXP_LIMIT => -KDA_DECAY_EXP_LIMIT,
        x if x > KDA_DECAY_EXP_LIMIT => KDA_DECAY_EXP_LIMIT,
        x => x,
    };
    exp_f32(x)
}

/// The forward clamp is locally constant outside this interval. At either
/// boundary retain the existing branch's one-sided active convention.
#[inline(always)]
pub(crate) fn kda_decay_exp_active(x: f32) -> bool {
    x >= -KDA_DECAY_EXP_LIMIT && x <= KDA_DECAY_EXP_LIMIT
}

#[inline(always)]
pub(crate) fn q_offset(_params: &CausalAttentionParams) -> u32 {
    0
}

#[inline(always)]
pub(crate) fn k_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim
}

#[inline(always)]
pub(crate) fn v_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 2
}

#[inline(always)]
pub(crate) fn g_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 3
}

#[inline(always)]
pub(crate) fn beta_offset(params: &CausalAttentionParams) -> u32 {
    params.embedding_dim * 4
}
