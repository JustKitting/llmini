use cuda_device::{thread, warp};

use crate::float_ptx::{abs_f32, max_f32};

use super::super::convert::{
    candidate_pair_absolute_error_and_payload, candidate_pair_errors_and_payload_with_inv_scale,
    local_scale_bits, nonzero_global_scale, nvfp4_inv_scale, scale_value,
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
    _group_mask: u32,
    _group_leader: u32,
    _lane_in_group: usize,
) -> (u8, u16) {
    // Lane i owns logical values 4*i..4*i+3. This preserves adjacent vector
    // loads while one four-lane subgroup covers the complete 16-value group.
    let lane_amax = max_f32(
        max_f32(abs_f32(value_0), abs_f32(value_1)),
        max_f32(abs_f32(value_2), abs_f32(value_3)),
    );
    // Every four lanes own one complete group. A uniform active mask keeps
    // all eight independent groups together during the two shuffle steps.
    let active: u32;
    unsafe {
        cuda_device::ptx_asm!("activemask.b32 %0;",out("=r")active,options(register_only));
    }
    let peer = warp::shuffle_xor_f32_sync(active, lane_amax, 1);
    let local = max_f32(lane_amax, peer);
    let group_amax = max_f32(local, warp::shuffle_xor_f32_sync(active, local, 2));
    // `group_amax` is identical in every lane of the four-lane subgroup. Let
    // every lane compute the identical scale instead of making the whole warp
    // execute a lane-leader branch and then broadcasting two values. SIMT
    // executes the scale instructions either way; this removes the divergent
    // branch and both shuffles without adding warp instructions.
    let scale_bits_six = local_scale_bits(group_amax, global_scale, scale_override, 6.0);
    let scale_six = scale_value(scale_bits_six);
    let inv_scale_six = nvfp4_inv_scale(scale_six, global_scale);

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

/// Optional activation-only selector. Both candidates use the existing E4M3
/// scale and E2M1 payload conversions; no extra represented operand is saved.
/// The training recipe minimizes L1 error, choosing six on exact ties.
#[inline(always)]
pub(crate) fn activation_grid_group_scale(
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
    if !crate::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX {
        return six_grid_group_scale(
            value_0,
            value_1,
            value_2,
            value_3,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
        );
    }
    let lane_amax = max_f32(
        max_f32(abs_f32(value_0), abs_f32(value_1)),
        max_f32(abs_f32(value_2), abs_f32(value_3)),
    );
    let active: u32;
    unsafe {
        cuda_device::ptx_asm!("activemask.b32 %0;", out("=r") active, options(register_only));
    }
    let local = max_f32(lane_amax, warp::shuffle_xor_f32_sync(active, lane_amax, 1));
    let group_amax = max_f32(local, warp::shuffle_xor_f32_sync(active, local, 2));
    let six = mae_candidate(
        value_0,
        value_1,
        value_2,
        value_3,
        group_amax,
        global_scale,
        scale_override,
        6.,
        active,
    );
    let four = mae_candidate(
        value_0,
        value_1,
        value_2,
        value_3,
        group_amax,
        global_scale,
        scale_override,
        4.,
        active,
    );
    if four.0 < six.0 {
        (four.1, four.2)
    } else {
        (six.1, six.2)
    }
}

#[inline(always)]
fn mae_candidate(
    value_0: f32,
    value_1: f32,
    value_2: f32,
    value_3: f32,
    group_amax: f32,
    global_scale: f32,
    scale_override: f32,
    grid_max: f32,
    active: u32,
) -> (f32, u8, u16) {
    let bits = local_scale_bits(group_amax, global_scale, scale_override, grid_max);
    let scale = scale_value(bits);
    let inverse = nvfp4_inv_scale(scale, global_scale);
    let (error_01, payload_01) =
        candidate_pair_absolute_error_and_payload(value_0, value_1, scale, global_scale, inverse);
    let (error_23, payload_23) =
        candidate_pair_absolute_error_and_payload(value_2, value_3, scale, global_scale, inverse);
    let lane_error = error_01 + error_23;
    let pair_error = lane_error + warp::shuffle_xor_f32_sync(active, lane_error, 1);
    let group_error = pair_error + warp::shuffle_xor_f32_sync(active, pair_error, 2);
    (
        group_error,
        bits as u8,
        payload_01 as u16 | ((payload_23 as u16) << 8),
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
