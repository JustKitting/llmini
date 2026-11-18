use cuda_device::{DisjointSlice, warp};

use crate::float_ptx::abs_f32;
use crate::nvfp4_cast::e4m3_value;
use crate::warp_reduce::half_warp_max_nonnegative_f32;

use super::super::super::convert::{nonzero_global_scale, nonzero_scale, nvfp4_inv_scale};
use super::super::rounding::{covering_scale_bits, stochastic_e2m1};
use super::super::{FP4_MAX, GROUP_SIZE, HADAMARD_DIM};

#[inline(always)]
pub(in super::super) fn ms_eden_scale_context(global_scale: f32) -> (f32, f32) {
    let safe_global_scale = nonzero_global_scale(global_scale);
    let fp4_global_inv_scale = nvfp4_inv_scale(FP4_MAX, safe_global_scale);
    (safe_global_scale, fp4_global_inv_scale)
}

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
    let (safe_global_scale, fp4_global_inv_scale) = ms_eden_scale_context(global_scale);
    ms_eden_pack_payload_precomputed_scale(
        value,
        out_fp4,
        out_scales,
        chunk,
        safe_global_scale,
        fp4_global_inv_scale,
        scale_override,
        scale_seed,
    );
}

#[inline(always)]
pub(in super::super) fn ms_eden_pack_payload_precomputed_scale(
    value: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    chunk: u32,
    safe_global_scale: f32,
    fp4_global_inv_scale: f32,
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
    let group_amax = half_warp_max_nonnegative_f32(abs_f32(value), group_mask);
    let scale_bits = covering_scale_bits(
        group_amax,
        safe_global_scale,
        group_amax * scale_override * fp4_global_inv_scale,
    );
    let scale = nonzero_scale(e4m3_value(scale_bits as u16));
    let inv_scale = nvfp4_inv_scale(scale, safe_global_scale);
    let x_scaled = value * inv_scale;
    let payload = stochastic_e2m1(x_scaled, scale_seed, chunk_base + lane);

    if lane == group_leader {
        unsafe {
            *out_scales.get_unchecked_mut(group as usize) = scale_bits;
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
