use cuda_device::{thread, warp};

use crate::float_ptx::{abs_f32, max_f32};
use crate::warp_reduce::{half_warp_max_nonnegative_f32, quarter_warp_sum_f32};

use super::super::convert::{
    candidate_pair_errors_and_payload_with_inv_scale, local_scale_bits, nonzero_global_scale,
    nvfp4_inv_scale, scale_value,
};

pub(crate) const GROUP_SIZE: usize = 16;
pub(crate) const GROUP_THREADS: usize = 8;

const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;

#[inline(always)]
pub(crate) fn four_six_lane() -> (usize, u32, u32) {
    let lane = warp::lane_id() as usize;
    let leader = lane & !(GROUP_THREADS - 1);
    let group_mask = 0xff_u32 << leader;
    (lane & (GROUP_THREADS - 1), group_mask, leader as u32)
}

#[inline(always)]
pub(crate) fn four_six_block_group() -> usize {
    let groups_per_block = thread::blockDim_x() as usize / GROUP_THREADS;
    thread::blockIdx_x() as usize * groups_per_block
        + thread::threadIdx_x() as usize / GROUP_THREADS
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
    value_lo: f32,
    value_hi: f32,
    global_scale: f32,
    scale_override: f32,
    group_mask: u32,
    group_leader: u32,
    lane_in_group: usize,
) -> (u8, u8) {
    // Lane i owns logical values i and i+8. Combining their error deltas here
    // is exactly the old XOR-8 reduction stage; the quarter-warp reduction
    // below retains the old XOR-4, XOR-2, XOR-1 order.
    let pair_amax = max_f32(abs_f32(value_lo), abs_f32(value_hi));
    let group_amax = half_warp_max_nonnegative_f32(pair_amax, group_mask);
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

    let (err_six_lo, err_six_hi, payload_six) = candidate_pair_errors_and_payload_with_inv_scale(
        value_lo,
        value_hi,
        scale_six,
        global_scale,
        inv_scale_six,
    );
    let (err_four_lo, err_four_hi, payload_four) = candidate_pair_errors_and_payload_with_inv_scale(
        value_lo,
        value_hi,
        scale_four,
        global_scale,
        inv_scale_four,
    );
    let pair_delta = (err_six_lo - err_four_lo) + (err_six_hi - err_four_hi);
    let error_delta = quarter_warp_sum_f32(pair_delta, group_mask);
    if error_delta <= 0.0 {
        (scale_bits_six as u8, payload_six)
    } else {
        (scale_bits_four as u8, payload_four)
    }
}

#[inline(always)]
pub(crate) fn four_six_payload_bytes(payload_pair: u8, group_mask: u32) -> (u8, u8) {
    let peer = warp::shuffle_xor_sync(group_mask, payload_pair as u32, 1) as u8;
    let first_half = (payload_pair & 0x0f) | ((peer & 0x0f) << 4);
    let second_half = (payload_pair >> 4) | (peer & 0xf0);
    (first_half, second_half)
}
