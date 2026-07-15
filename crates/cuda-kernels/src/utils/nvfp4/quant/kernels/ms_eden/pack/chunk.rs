#![expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]

use cuda_device::{DisjointSlice, thread, warp};

use crate::float_ptx::abs_f32;
use crate::warp_reduce::warp_max_nonnegative_f32;

use super::super::{AMAX_WARPS_PER_BLOCK, HADAMARD_DIM};
use super::hadamard::hadamard_transform_lane;
use super::payload::ms_eden_pack_payload;

#[inline(always)]
pub(in super::super) fn ms_eden_pack_chunk(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    out_chunk_amax: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    dst_row_len: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let (value, lane) =
        pack_chunk_value(input, out_global_scales, chunk, dst_row_len, global_scale);

    let chunk_amax = warp_max_nonnegative_f32(abs_f32(value));
    if lane == 0 {
        unsafe {
            *out_chunk_amax.get_unchecked_mut(chunk as usize) = chunk_amax;
        }
    }

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[inline(always)]
pub(in super::super) fn ms_eden_pack_chunk_no_chunk_amax(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    dst_row_len: u32,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let (value, _) = pack_chunk_value(input, out_global_scales, chunk, dst_row_len, global_scale);

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[inline(always)]
pub(in super::super) fn ms_eden_pack_chunk_no_chunk_amax_row(
    input: f32,
    out_fp4: &mut DisjointSlice<'_, u8>,
    out_scales: &mut DisjointSlice<'_, u8>,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    row: u32,
    first_chunk_in_row: bool,
    global_scale: f32,
    scale_override: f32,
    scale_seed: u32,
) {
    let lane = warp::lane_id();
    let value = hadamard_transform_lane(input, lane);

    if first_chunk_in_row && lane == 0 {
        unsafe {
            *out_global_scales.get_unchecked_mut(row as usize) = global_scale;
        }
    }

    ms_eden_pack_payload(
        value,
        out_fp4,
        out_scales,
        chunk,
        global_scale,
        scale_override,
        scale_seed,
    );
}

#[inline(always)]
pub(in super::super) fn pack_chunk() -> u32 {
    thread::blockIdx_x() * AMAX_WARPS_PER_BLOCK + thread::threadIdx_x() / 32
}

#[inline(always)]
fn pack_chunk_value(
    input: f32,
    out_global_scales: &mut DisjointSlice<'_, f32>,
    chunk: u32,
    dst_row_len: u32,
    global_scale: f32,
) -> (f32, u32) {
    let lane = warp::lane_id();
    let element = chunk * HADAMARD_DIM + lane;
    let row = element / dst_row_len;
    let value = hadamard_transform_lane(input, lane);

    if element == row * dst_row_len {
        unsafe {
            *out_global_scales.get_unchecked_mut(row as usize) = global_scale;
        }
    }

    (value, lane)
}
