use cuda_device::ptx_asm;

use crate::float_ptx::rcp_approx_f32;
use crate::nvfp4_cast::{e2m1_value, e4m3_value};

const NVFP4_DENOM_EPS: f32 = 1.0e-20;

#[inline(always)]
pub(crate) fn nonzero_global_scale(global_scale: f32) -> f32 {
    if global_scale == 0.0 {
        1.0
    } else {
        global_scale
    }
}

#[inline(always)]
pub(crate) fn nonzero_scale(scale: f32) -> f32 {
    if scale == 0.0 { 1.0 } else { scale }
}

#[inline(always)]
pub(crate) fn nvfp4_inv_scale(scale: f32, global_scale: f32) -> f32 {
    rcp_approx_f32(nonzero_scale(scale) * nonzero_global_scale(global_scale) + NVFP4_DENOM_EPS)
}

#[inline(always)]
pub(crate) fn candidate_pair_errors_and_payload_with_inv_scale(
    value_lo: f32,
    value_hi: f32,
    scale: f32,
    global_scale: f32,
    inv_scale: f32,
) -> (f32, f32, u8) {
    let global_scale = nonzero_global_scale(global_scale);
    let dequant_scale = scale * global_scale;
    let packed = cvt_rn_satfinite_e2m1x2_f32(value_hi * inv_scale, value_lo * inv_scale);
    let payload_lo = packed & 0x0f;
    let payload_hi = packed >> 4;
    let diff_lo = value_lo - e2m1_value(payload_lo) * dequant_scale;
    let diff_hi = value_hi - e2m1_value(payload_hi) * dequant_scale;
    (diff_lo * diff_lo, diff_hi * diff_hi, packed)
}

#[inline(always)]
pub(crate) fn local_scale_bits(
    group_amax: f32,
    global_scale: f32,
    scale_override: f32,
    grid_max: f32,
) -> u16 {
    let global_scale = nonzero_global_scale(global_scale);
    let value =
        group_amax * scale_override * rcp_approx_f32(grid_max * global_scale + NVFP4_DENOM_EPS);
    let packed: u16;

    unsafe {
        ptx_asm!(
            "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;",
            out("=h") packed,
            in("f") 0.0f32,
            in("f") value,
            options(register_only),
        );
    }
    packed
}

#[inline(always)]
pub(crate) fn scale_value(bits: u16) -> f32 {
    e4m3_value(bits)
}

#[inline(always)]
pub(crate) fn cvt_rn_satfinite_e2m1x2_f32(hi: f32, lo: f32) -> u8 {
    let packed: u16;
    unsafe {
        ptx_asm!(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u16.u8 %0, tmp; }",
            out("=h") packed,
            in("f") hi,
            in("f") lo,
            options(register_only),
        );
    }
    packed as u8
}

#[inline(always)]
pub(super) fn cvt_rn_satfinite_e4m3x2_f32(hi: f32, lo: f32) -> u8 {
    let packed: u16;
    unsafe {
        ptx_asm!(
            "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;",
            out("=h") packed,
            in("f") hi,
            in("f") lo,
            options(register_only),
        );
    }
    packed as u8
}
