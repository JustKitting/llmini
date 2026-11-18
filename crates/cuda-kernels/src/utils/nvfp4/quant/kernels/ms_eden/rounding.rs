//! Conditional stochastic FP4 rounding after the shared sign/Hadamard rotation.
use super::super::convert::{cvt_rn_satfinite_e4m3x2_f32, nvfp4_inv_scale};
use super::random::hash_u32;
use crate::float_ptx::max_f32;
use crate::nvfp4_cast::e4m3_value;

#[inline(always)]
pub(super) fn covering_scale_bits(amax: f32, global: f32, preferred: f32) -> u8 {
    // Nearest or downward scale rounding can put the block maximum above 6.
    // Choose a representable E4M3 scale covering the whole block instead.
    let required = amax * nvfp4_inv_scale(6.0, global);
    let desired = max_f32(required, preferred);
    let mut bits = cvt_rn_satfinite_e4m3x2_f32(0.0, desired);
    if e4m3_value(bits as u16) < desired && bits < 126 {
        bits += 1;
    }
    bits
}

#[inline(always)]
pub(super) fn stochastic_e2m1(value: f32, seed: u32, index: u32) -> u8 {
    let bits = value.to_bits();
    let a = f32::from_bits(bits & 0x7fff_ffff);
    let sign = ((bits >> 28) & 8) as u8;
    if a >= 6.0 {
        return sign | 7;
    }
    let (lower, gap, code) = if a < 0.5 {
        (0.0, 0.5, 0u8)
    } else if a < 1.0 {
        (0.5, 0.5, 1)
    } else if a < 1.5 {
        (1.0, 0.5, 2)
    } else if a < 2.0 {
        (1.5, 0.5, 3)
    } else if a < 3.0 {
        (2.0, 1.0, 4)
    } else if a < 4.0 {
        (3.0, 1.0, 5)
    } else {
        (4.0, 2.0, 6)
    };
    let probability = (a - lower) / gap;
    // Use the supplied, operand-separated seed and logical output coordinate.
    // Probability discretization is at most one part in 2^32.
    let cutoff = (probability * 4_294_967_296.0) as u32;
    let random = hash_u32(seed ^ index.wrapping_mul(0x9e37_79b9));
    sign | (code + (random < cutoff) as u8)
}

#[inline(always)]
pub(super) fn stochastic_pair(high: f32, low: f32, seed: u32, base: u32) -> u8 {
    stochastic_e2m1(low, seed, base) | (stochastic_e2m1(high, seed, base + 1) << 4)
}
