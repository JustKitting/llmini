use cuda_device::DisjointSlice;

use super::super::ApplyRopeParams;
use super::pair::{QkvPair, read_pair};
use crate::float_ptx::fma_f32;
use crate::position::{rotation_factors, stationary_dim};

#[inline(always)]
pub(super) fn rotate_section(
    qkv: &mut DisjointSlice<f32>,
    batch: u32,
    token: u32,
    head: u32,
    dim: u32,
    section_offset: u32,
    params: &ApplyRopeParams,
) -> QkvPair {
    let pair = read_pair(qkv, batch, token, head, dim, section_offset, params);
    if stationary_dim(dim, params.head_dim) {
        return pair;
    }
    let (sin, cos) = rotation_factors(token, dim, params.head_dim);
    let rotated_even = fma_f32(-pair.odd, sin, pair.even * cos);
    let rotated_odd = fma_f32(pair.odd, cos, pair.even * sin);

    unsafe {
        let ptr = qkv.as_mut_ptr();
        *ptr.add(pair.even_index) = rotated_even;
        *ptr.add(pair.odd_index) = rotated_odd;
    }

    QkvPair {
        even_index: pair.even_index,
        even: rotated_even,
        odd_index: pair.odd_index,
        odd: rotated_odd,
    }
}
