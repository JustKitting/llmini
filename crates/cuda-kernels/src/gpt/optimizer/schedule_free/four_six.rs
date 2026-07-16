use cuda_device::DisjointSlice;

use crate::nvfp4_quant::kernels::four_six::helpers::{
    GROUP_SIZE, GROUP_THREADS, four_six_block_group, four_six_global_scale, four_six_group_scale,
    four_six_lane, four_six_payload_bytes,
};

use super::SCALE_OVERRIDE;
use super::value::schedule_value;

pub(super) fn schedule_free_four_six_body(
    z_master: &[f32],
    x_master: &[f32],
    amax: &[f32],
    out_fp4: &mut DisjointSlice<u8>,
    out_scales: &mut DisjointSlice<u8>,
    out_global_scale: &mut DisjointSlice<f32>,
    beta: f32,
) {
    let (lane_in_group, group_mask, group_leader) = four_six_lane();
    let group = four_six_block_group();

    if group < out_scales.len() {
        let base = group * GROUP_SIZE;
        let tensor_amax = amax[0];
        let global_scale = four_six_global_scale(tensor_amax, SCALE_OVERRIDE);
        unsafe {
            if group == 0 && lane_in_group == 0 {
                *out_global_scale.get_unchecked_mut(0) = global_scale;
            }

            let value_lo = schedule_value(z_master, x_master, beta, (base + lane_in_group) as u32);
            let value_hi = schedule_value(
                z_master,
                x_master,
                beta,
                (base + lane_in_group + GROUP_THREADS) as u32,
            );
            let (scale_bits, payload_pair) = four_six_group_scale(
                value_lo,
                value_hi,
                global_scale,
                SCALE_OVERRIDE,
                group_mask,
                group_leader,
                lane_in_group,
            );

            if lane_in_group == 0 {
                *out_scales.get_unchecked_mut(group) = scale_bits;
            }
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, group_mask);
            if lane_in_group.is_multiple_of(2) {
                *out_fp4.get_unchecked_mut(base / 2 + lane_in_group / 2) = payload_lo;
                *out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane_in_group / 2) =
                    payload_hi;
            }
        }
    }
}
