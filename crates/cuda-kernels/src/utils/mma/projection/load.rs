use super::args::Nvfp4ProjectionParams;
use super::load_bytes::{E4M3_ONE_PACKED4, load_packed8, load_scale4};

const SCALE_GROUP: u32 = 16;

#[inline(always)]
pub(super) fn load_a_fragments(
    bytes: &[u8],
    tile_row: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
) -> [u32; 4] {
    [
        load_a_fragment(bytes, tile_row, k_base, group, thread_in_group, params, 0),
        load_a_fragment(bytes, tile_row, k_base, group, thread_in_group, params, 1),
        load_a_fragment(bytes, tile_row, k_base, group, thread_in_group, params, 2),
        load_a_fragment(bytes, tile_row, k_base, group, thread_in_group, params, 3),
    ]
}

#[inline(always)]
pub(super) fn load_b_fragments(
    bytes: &[u8],
    tile_col: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
) -> [u32; 2] {
    [
        load_b_fragment(bytes, tile_col, k_base, group, thread_in_group, params, 0),
        load_b_fragment(bytes, tile_col, k_base, group, thread_in_group, params, 1),
    ]
}

#[inline(always)]
pub(super) fn load_a_scale4(
    scales: &[u8],
    tile_row: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let row = tile_row + group + if thread_in_group == 1 { 8 } else { 0 };
    if row < params.token_count {
        load_scale4(
            scales,
            ((row * params.input_dim + k_base) / SCALE_GROUP) as usize,
        )
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
pub(super) fn load_b_scale4(
    scales: &[u8],
    tile_col: u32,
    k_base: u32,
    group: u32,
    params: &Nvfp4ProjectionParams,
) -> u32 {
    let col = tile_col + group;
    if col < params.output_dim {
        let scale_base = col * (params.input_dim / SCALE_GROUP) + k_base / SCALE_GROUP;
        crate::nvfp4::csf::scale4(scales, scale_base as usize)
    } else {
        E4M3_ONE_PACKED4
    }
}

#[inline(always)]
fn load_a_fragment(
    bytes: &[u8],
    tile_row: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
    register: u32,
) -> u32 {
    let row = tile_row + group + if register & 1 == 0 { 0 } else { 8 };
    let col = k_base + thread_in_group * 8 + if register < 2 { 0 } else { 32 };

    if row < params.token_count && col + 7 < params.input_dim {
        load_packed8(
            bytes,
            row as usize * params.input_dim as usize + col as usize,
        )
    } else {
        0
    }
}

#[inline(always)]
fn load_b_fragment(
    bytes: &[u8],
    tile_col: u32,
    k_base: u32,
    group: u32,
    thread_in_group: u32,
    params: &Nvfp4ProjectionParams,
    register: u32,
) -> u32 {
    let col = tile_col + group;
    let row = k_base + thread_in_group * 8 + if register == 0 { 0 } else { 32 };

    if col < params.output_dim && row + 7 < params.input_dim {
        load_packed8(
            bytes,
            col as usize * params.input_dim as usize + row as usize,
        )
    } else {
        0
    }
}
