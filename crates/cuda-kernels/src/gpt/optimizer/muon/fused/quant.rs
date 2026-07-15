use cuda_device::SharedArray;

use super::super::super::threads::WARPS_PER_BLOCK;
use super::super::super::work_grid::WorkGrid;

mod encode;
mod reduce_scale;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn quantize_updated_master(
    x: *const f32,
    block_amax: *mut f32,
    out_fp4: *mut u8,
    out_scales: *mut u8,
    out_global_scale: *mut f32,
    len: u32,
    warp_sums: &mut SharedArray<f32, { WARPS_PER_BLOCK as usize }>,
    work: WorkGrid,
) {
    reduce_scale::reduce_global_scale(block_amax, out_global_scale, warp_sums, work);
    encode::encode_four_six(x, out_fp4, out_scales, out_global_scale, len, work);
}
