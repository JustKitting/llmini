use cuda_device::{
    SharedArray,
    atomic::{AtomicOrdering, DeviceAtomicU32},
};

use super::super::symexp_lin::ScalePointers;
use super::super::threads::WARPS_PER_BLOCK;
use super::value::{checked_abs_schedule_value, schedule_value};
use crate::amax::{amax4_f32, max4_f32};
use crate::block_reduce::block_max_leader_f32;
use crate::float_ptx::max_f32;
use crate::nvfp4_quant::kernels::row_amax::{
    TENSOR_AMAX_VALUES_PER_BLOCK, tensor_amax_chunk_indices,
};

pub(super) fn schedule_free_amax_body(
    z_master: &[f32],
    x_master: &[f32],
    output_amax: &[f32],
    beta: f32,
    symexp_lin_beta: f32,
    scales: ScalePointers,
    len: u32,
) {
    static mut TENSOR_AMAX: SharedArray<f32, { WARPS_PER_BLOCK as usize }> = SharedArray::UNINIT;

    let scalars = scales.scalars(beta, symexp_lin_beta);
    let (_, lane, warp_in_block, base, i0, i1, i2, i3, i4, i5, i6, i7) =
        tensor_amax_chunk_indices();
    let local_amax = if base + TENSOR_AMAX_VALUES_PER_BLOCK <= len {
        max_f32(
            amax4_f32(
                schedule_value(z_master, x_master, beta, scalars, i0),
                schedule_value(z_master, x_master, beta, scalars, i1),
                schedule_value(z_master, x_master, beta, scalars, i2),
                schedule_value(z_master, x_master, beta, scalars, i3),
            ),
            amax4_f32(
                schedule_value(z_master, x_master, beta, scalars, i4),
                schedule_value(z_master, x_master, beta, scalars, i5),
                schedule_value(z_master, x_master, beta, scalars, i6),
                schedule_value(z_master, x_master, beta, scalars, i7),
            ),
        )
    } else {
        max_f32(
            max4_f32(
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i0, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i1, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i2, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i3, len),
            ),
            max4_f32(
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i4, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i5, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i6, len),
                checked_abs_schedule_value(z_master, x_master, beta, scalars, i7, len),
            ),
        )
    };

    if let Some(cta_amax) =
        unsafe { block_max_leader_f32(&mut TENSOR_AMAX, local_amax, lane, warp_in_block) }
    {
        let output = unsafe { &*(output_amax.as_ptr() as *const DeviceAtomicU32) };
        output.fetch_max(cta_amax.to_bits(), AtomicOrdering::Relaxed);
    }
}
