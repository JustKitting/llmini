use crate::f16_tc_matmul::convert::load_f32x2_global;
use crate::float_ptx::abs_f32;

use super::super::symexp_lin::{ScalePointers, mismatch_forward_scaled};

#[inline(always)]
pub(super) fn schedule_value(
    z_master: &[f32],
    x_master: &[f32],
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
    index: u32,
) -> f32 {
    let i = index as usize;
    let z = z_master[i];
    let x = x_master[i];
    let (exponential, linear, curvature) = scales.values(beta);
    mismatch_forward_scaled(
        z + beta * (x - z),
        symexp_lin_beta,
        exponential,
        linear,
        curvature,
    )
}

#[inline(always)]
pub(super) fn schedule_value_pair(
    z_master: &[f32],
    x_master: &[f32],
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
    index: usize,
) -> (f32, f32) {
    let (z_lo, z_hi) = load_f32x2_global(z_master.as_ptr(), index);
    let (x_lo, x_hi) = load_f32x2_global(x_master.as_ptr(), index);
    let (exponential, linear, curvature) = scales.values(beta);
    (
        mismatch_forward_scaled(
            z_lo + beta * (x_lo - z_lo),
            symexp_lin_beta,
            exponential,
            linear,
            curvature,
        ),
        mismatch_forward_scaled(
            z_hi + beta * (x_hi - z_hi),
            symexp_lin_beta,
            exponential,
            linear,
            curvature,
        ),
    )
}

#[inline(always)]
pub(super) fn schedule_value_quad(
    z_master: &[f32],
    x_master: &[f32],
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
    index: usize,
) -> (f32, f32, f32, f32) {
    let (value_0, value_1) =
        schedule_value_pair(z_master, x_master, beta, symexp_lin_beta, scales, index);
    let (value_2, value_3) =
        schedule_value_pair(z_master, x_master, beta, symexp_lin_beta, scales, index + 2);
    (value_0, value_1, value_2, value_3)
}

#[inline(always)]
pub(super) fn checked_abs_schedule_value(
    z_master: &[f32],
    x_master: &[f32],
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
    index: u32,
    len: u32,
) -> f32 {
    if index < len {
        abs_f32(schedule_value(
            z_master,
            x_master,
            beta,
            symexp_lin_beta,
            scales,
            index,
        ))
    } else {
        0.0
    }
}
