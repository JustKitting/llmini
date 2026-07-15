use cuda_device::{thread, warp};

use crate::float_ptx::abs_f32;
use crate::warp_reduce::{half_warp_max_f32, half_warp_sum_f32};

use super::super::convert::{
    candidate_error_and_payload_with_inv_scale, local_scale_bits, nonzero_global_scale,
    nvfp4_inv_scale, scale_value,
};

pub(crate) const GROUP_SIZE: usize = 16;

const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;

#[inline(always)]
pub(crate) fn four_six_lane() -> (usize, u32, u32) {
    let lane = warp::lane_id() as usize;
    let group_mask = if lane < GROUP_SIZE {
        0x0000_ffff
    } else {
        0xffff_0000
    };
    (lane & 0x0f, group_mask, (lane & !0x0f) as u32)
}

#[inline(always)]
pub(crate) fn four_six_block_group() -> usize {
    let groups_per_block = thread::blockDim_x() as usize / GROUP_SIZE;
    thread::blockIdx_x() as usize * groups_per_block + thread::threadIdx_x() as usize / GROUP_SIZE
}

#[inline(always)]
pub(crate) fn four_six_global_scale(tensor_amax: f32, scale_override: f32) -> f32 {
    nonzero_global_scale(if tensor_amax == 0.0 {
        1.0
    } else {
        tensor_amax * scale_override / (FP8_MAX_FOUR_SIX * FP4_MAX)
    })
}

#[inline(always)]
pub(crate) fn four_six_group_scale(
    value: f32,
    global_scale: f32,
    scale_override: f32,
    group_mask: u32,
    group_leader: u32,
    lane_in_group: usize,
) -> (u8, u8) {
    let group_amax = half_warp_max_f32(abs_f32(value), group_mask);
    let mut scale_bits_six = 0u16;
    let mut scale_bits_four = 0u16;
    let mut scale_six = 0.0;
    let mut scale_four = 0.0;
    let mut inv_scale_six = 0.0;
    let mut inv_scale_four = 0.0;

    if lane_in_group == 0 {
        scale_bits_six = local_scale_bits(group_amax, global_scale, scale_override, 6.0);
        scale_bits_four = local_scale_bits(group_amax, global_scale, scale_override, 4.0);
        scale_six = scale_value(scale_bits_six);
        scale_four = scale_value(scale_bits_four);
        inv_scale_six = nvfp4_inv_scale(scale_six, global_scale);
        inv_scale_four = nvfp4_inv_scale(scale_four, global_scale);
    }

    // Every caller stores scale bits only from the group leader, where both
    // candidates were produced. The candidate scales remain group-wide.
    scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
    scale_four = warp::shuffle_f32_sync(group_mask, scale_four, group_leader);
    inv_scale_six = warp::shuffle_f32_sync(group_mask, inv_scale_six, group_leader);
    inv_scale_four = warp::shuffle_f32_sync(group_mask, inv_scale_four, group_leader);

    let (local_err_six, payload_six) =
        candidate_error_and_payload_with_inv_scale(value, scale_six, global_scale, inv_scale_six);
    let (local_err_four, payload_four) =
        candidate_error_and_payload_with_inv_scale(value, scale_four, global_scale, inv_scale_four);
    let err_six = half_warp_sum_f32(local_err_six, group_mask);
    let err_four = half_warp_sum_f32(local_err_four, group_mask);
    if err_six <= err_four {
        (scale_bits_six as u8, payload_six)
    } else {
        (scale_bits_four as u8, payload_four)
    }
}

#[inline(always)]
pub(crate) fn four_six_payload_byte(payload: u8, group_mask: u32) -> u8 {
    let peer = warp::shuffle_xor_sync(group_mask, payload as u32, 1) as u8;
    payload | (peer << 4)
}
