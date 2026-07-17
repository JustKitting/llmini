use cuda_device::{thread, warp};

use crate::float_ptx::{abs_f32, max_f32};
use crate::warp_reduce::half_warp_max_nonnegative_f32;

use super::super::convert::{
    candidate_pair_errors_and_payload_with_inv_scale, local_scale_bits, nonzero_global_scale,
    nvfp4_inv_scale, scale_value,
};

pub(crate) const GROUP_SIZE: usize = 16;
pub(crate) const GROUP_THREADS: usize = 4;

const FP4_MAX: f32 = 6.0;
const FP8_MAX_FOUR_SIX: f32 = 256.0;

#[inline(always)]
pub(crate) fn four_six_lane() -> (usize, u32, u32) {
    let lane = warp::lane_id() as usize;
    let leader = lane & !(GROUP_THREADS - 1);
    let group_mask = 0x0f_u32 << leader;
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
pub(crate) fn six_grid_group_scale(
    value_0: f32,
    value_1: f32,
    value_2: f32,
    value_3: f32,
    global_scale: f32,
    scale_override: f32,
    group_mask: u32,
    group_leader: u32,
    lane_in_group: usize,
) -> (u8, u16) {
    // Lane i owns logical values 4*i..4*i+3. This preserves adjacent vector
    // loads while one four-lane subgroup covers the complete 16-value group.
    let lane_amax = max_f32(
        max_f32(abs_f32(value_0), abs_f32(value_1)),
        max_f32(abs_f32(value_2), abs_f32(value_3)),
    );
    let group_amax = half_warp_max_nonnegative_f32(lane_amax, group_mask);
    let mut scale_bits_six = 0u16;
    let mut scale_six = 0.0;
    let mut inv_scale_six = 0.0;

    if lane_in_group == 0 {
        scale_bits_six = local_scale_bits(group_amax, global_scale, scale_override, 6.0);
        scale_six = scale_value(scale_bits_six);
        inv_scale_six = nvfp4_inv_scale(scale_six, global_scale);
    }

    // Every caller stores scale bits only from the group leader. The selected
    // scale and reciprocal remain group-wide.
    scale_six = warp::shuffle_f32_sync(group_mask, scale_six, group_leader);
    inv_scale_six = warp::shuffle_f32_sync(group_mask, inv_scale_six, group_leader);

    let (_, _, payload_six_01) = candidate_pair_errors_and_payload_with_inv_scale(
        value_0,
        value_1,
        scale_six,
        global_scale,
        inv_scale_six,
    );
    let (_, _, payload_six_23) = candidate_pair_errors_and_payload_with_inv_scale(
        value_2,
        value_3,
        scale_six,
        global_scale,
        inv_scale_six,
    );
    (
        scale_bits_six as u8,
        payload_six_01 as u16 | ((payload_six_23 as u16) << 8),
    )
}

#[inline(always)]
pub(crate) unsafe fn store_four_six_payload_word(
    out_fp4: *mut u8,
    group_base: usize,
    lane_in_group: usize,
    payload_word: u16,
) {
    unsafe {
        *out_fp4.cast::<u16>().add(group_base / 4 + lane_in_group) = payload_word;
    }
}
