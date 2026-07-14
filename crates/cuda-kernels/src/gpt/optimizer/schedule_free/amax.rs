use cuda_device::{DisjointSlice, SharedArray};

use crate::amax::{amax4_f32, max4_f32};
use crate::block_reduce::block_max_store_f32;
use crate::float_ptx::max_f32;
use crate::nvfp4_quant::kernels::row_amax::{
    TENSOR_AMAX_VALUES_PER_BLOCK, tensor_amax_chunk_indices,
};

use super::super::threads::WARPS_PER_BLOCK;
use super::value::{checked_abs_schedule_value, schedule_value};

pub(super) fn schedule_free_chunk_amax_body(
    z_master: &[f32],
    x_master: &[f32],
    out: &mut DisjointSlice<f32>,
    beta: f32,
    len: u32,
) {
    static mut TENSOR_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    let (chunk, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
        tensor_amax_chunk_indices();
    let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= len {
        max_f32(
            amax4_f32(
                schedule_value(z_master, x_master, beta, i0),
                schedule_value(z_master, x_master, beta, i1),
                schedule_value(z_master, x_master, beta, i2),
                schedule_value(z_master, x_master, beta, i3),
            ),
            amax4_f32(
                schedule_value(z_master, x_master, beta, i4),
                schedule_value(z_master, x_master, beta, i5),
                schedule_value(z_master, x_master, beta, i6),
                schedule_value(z_master, x_master, beta, i7),
            ),
        )
    } else {
        max_f32(
            max4_f32(
                checked_abs_schedule_value(z_master, x_master, beta, i0, len),
                checked_abs_schedule_value(z_master, x_master, beta, i1, len),
                checked_abs_schedule_value(z_master, x_master, beta, i2, len),
                checked_abs_schedule_value(z_master, x_master, beta, i3, len),
            ),
            max4_f32(
                checked_abs_schedule_value(z_master, x_master, beta, i4, len),
                checked_abs_schedule_value(z_master, x_master, beta, i5, len),
                checked_abs_schedule_value(z_master, x_master, beta, i6, len),
                checked_abs_schedule_value(z_master, x_master, beta, i7, len),
            ),
        )
    };

    block_max_store_f32!(TENSOR_AMAX, out[chunk], local_amax, lane, warp_in_block);
}
