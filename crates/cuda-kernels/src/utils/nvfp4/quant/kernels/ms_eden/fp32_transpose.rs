use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::body::fp32_transpose_to_nvfp4_ms_eden_body;
use super::pack::guarded_pack_chunk;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        x: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_chunk_amax: DisjointSlice<f32>,
        global_scale: &[f32],
        chunk_count: u32,
        source_rows: u32,
        source_cols: u32,
        dst_row_len: u32,
        scale_override: f32,
        sign_seed: u32,
        scale_seed: u32,
    ) {
        guarded_pack_chunk!(chunk, chunk_count);

        fp32_transpose_to_nvfp4_ms_eden_body(
            x,
            &mut out_fp4,
            &mut out_scales,
            core::ptr::null_mut(),
            &mut out_chunk_amax,
            chunk,
            source_rows,
            dst_row_len,
            source_cols,
            global_scale[0],
            scale_override,
            sign_seed,
            scale_seed,
        );
    }
}
