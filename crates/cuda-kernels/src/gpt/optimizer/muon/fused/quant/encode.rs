use cuda_device::thread;

use crate::device_ptr::read_f32;
use crate::f16_tc_matmul::cta_tile::CTA_THREADS;
use crate::nvfp4_quant::kernels::four_six::helpers::{
    GROUP_SIZE, GROUP_THREADS, four_six_group_scale, four_six_lane, four_six_payload_bytes,
};

use super::super::super::super::work_grid::WorkGrid;

const SCALE_OVERRIDE: f32 = 1.0;

pub(crate) fn encode_four_six(
    x: *const f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    out_global_scale: *mut f32,
    len: u32,
    work: WorkGrid,
) {
    let groups_per_block = CTA_THREADS / GROUP_THREADS as u32;
    let group_count = len / GROUP_SIZE as u32;
    let group_stride = work.blocks() * groups_per_block;
    let mut group = work.block() * groups_per_block + thread::threadIdx_x() / GROUP_THREADS as u32;
    let global_scale = unsafe { *out_global_scale };

    while group < group_count {
        encode_group(x, out_fp4, out_scales, global_scale, group);
        group += group_stride;
    }
}

fn encode_group(
    x: *const f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    global_scale: f32,
    group: u32,
) {
    let (lane_in_group, group_mask, group_leader) = four_six_lane();
    let base = group * GROUP_SIZE as u32;
    let value_lo = read_f32(x, base + lane_in_group as u32);
    let value_hi = read_f32(x, base + lane_in_group as u32 + GROUP_THREADS as u32);
    let (scale_bits, payload_pair) = four_six_group_scale(
        value_lo,
        value_hi,
        global_scale,
        SCALE_OVERRIDE,
        group_mask,
        group_leader,
        lane_in_group,
    );
    let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, group_mask);

    unsafe {
        if lane_in_group == 0 {
            *out_scales.add(group as usize) = scale_bits;
        }
        if lane_in_group.is_multiple_of(2) {
            *out_fp4.add((base / 2 + lane_in_group as u32 / 2) as usize) = payload_lo;
            *out_fp4
                .add((base / 2 + GROUP_THREADS as u32 / 2 + lane_in_group as u32 / 2) as usize) =
                payload_hi;
        }
    }
}
