use cuda_device::{DisjointSlice, warp};

use crate::float_ptx::abs_f32;
use crate::nvfp4_cast::{e2m1_value, e4m3_value};
use crate::warp_reduce::{half_warp_max_nonnegative_f32, half_warp_sum_f32};

use super::super::super::convert::{
    cvt_rn_satfinite_e2m1x2_f32, cvt_rn_satfinite_e4m3x2_f32, nonzero_global_scale, nonzero_scale,
    nvfp4_inv_scale,
};
use super::super::{FP4_MAX, GROUP_SIZE, HADAMARD_DIM};
use super::scale::stochastic_e4m3_scale;

#[inline(always)]
pub(super) fn ms_eden_pack_payload(
    value: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    chunk: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let chunk_base = chunk * HADAMARD_DIM;
    let lane_in_group = lane & 0x0f;
    let group_mask = if lane < GROUP_SIZE {
        0x0000_ffff
    } else {
        0xffff_0000
    };
    let group_leader = lane & !0x0f;
    let group = chunk * 2 + lane / GROUP_SIZE;
    let safe_global_scale = nonzero_global_scale(global_scale);
    let group_amax = half_warp_max_nonnegative_f32(abs_f32(value), group_mask);
    let scale_bits = cvt_rn_satfinite_e4m3x2_f32(
        0.0,
        group_amax * scale_override * nvfp4_inv_scale(FP4_MAX, safe_global_scale),
    );
    let scale = nonzero_scale(e4m3_value(scale_bits as u16));
    let inv_scale = nvfp4_inv_scale(scale, safe_global_scale);
    let x_scaled = value * inv_scale;
    let payload = cvt_rn_satfinite_e2m1x2_f32(0.0, x_scaled) & 0x0f;
    let fp4_value = e2m1_value(payload);

    let num = half_warp_sum_f32(x_scaled * x_scaled, group_mask);
    let denom = half_warp_sum_f32(x_scaled * fp4_value, group_mask);
    let correction = if denom == 0.0 { 1.0 } else { num / denom };
    let corrected_scale = nonzero_scale(scale * correction);
    let rounded_scale_bits = stochastic_e4m3_scale(corrected_scale, scale_seed, group);

    if lane == group_leader {
        unsafe {
            *out_scales.get_unchecked_mut(group as usize) = rounded_scale_bits;
        }
    }

    let peer_payload = warp::shuffle_xor_sync(0xffff_ffff, payload as u32, 1) as u8;
    if lane_in_group.is_multiple_of(2) {
        let byte = chunk_base / 2 + (lane / GROUP_SIZE) * (GROUP_SIZE / 2) + lane_in_group / 2;
        unsafe {
            *out_fp4.get_unchecked_mut(byte as usize) = payload | (peer_payload << 4);
        }
    }
}
