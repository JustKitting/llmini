use crate::float_ptx::rcp_approx_f32;
use crate::nvfp4_cast::e4m3_value;

use super::super::super::convert::cvt_rn_satfinite_e4m3x2_f32;
use super::super::random::random_unit_f32;

#[inline(always)]
pub(super) fn stochastic_e4m3_scale(value: f32, seed: u32, group: u32) -> u8 {
    let curr_bits = cvt_rn_satfinite_e4m3x2_f32(0.0, value);
    let curr = e4m3_value(curr_bits as u16);
    let prev_bits = curr_bits.saturating_sub(1);
    let next_bits = curr_bits.saturating_add(1);
    let prev = e4m3_value(prev_bits as u16);
    let next = e4m3_value(next_bits as u16);
    let up = if curr > value { curr } else { next };
    let down = if curr > value { prev } else { curr };
    let up_bits = if curr > value { curr_bits } else { next_bits };
    let down_bits = if curr > value { prev_bits } else { curr_bits };
    let denom = up - down;
    let prob_up = if denom == 0.0 {
        0.0
    } else {
        (value - down) * rcp_approx_f32(denom)
    };

    if random_unit_f32(seed, group) < prob_up {
        up_bits
    } else {
        down_bits
    }
}
