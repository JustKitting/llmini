use cuda_device::DisjointSlice;

use crate::nvfp4_quant::kernels::four_six::helpers::{
    GROUP_SIZE, four_six_block_group, four_six_global_scale, four_six_lane, six_grid_group_scale,
    store_four_six_payload_word,
};

use super::super::symexp_lin::ScalePointers;
use super::SCALE_OVERRIDE;
use super::value::schedule_value_quad;

pub(super) fn schedule_free_four_six_body(
    z_master: &[f32],
    x_master: &[f32],
    amax: &[f32],
    out_fp4: &mut DisjointSlice<u8>,
    out_scales: &mut DisjointSlice<u8>,
    out_global_scale: &mut DisjointSlice<f32>,
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
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

            let (value_0, value_1, value_2, value_3) = schedule_value_quad(
                z_master,
                x_master,
                beta,
                symexp_lin_beta,
                scales,
                base + 4 * lane_in_group,
            );
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                SCALE_OVERRIDE,
                group_mask,
                group_leader,
                lane_in_group,
            );

            if lane_in_group == 0 {
                *out_scales.get_unchecked_mut(group) = scale_bits;
            }
            store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane_in_group, payload_word);
        }
    }
}
