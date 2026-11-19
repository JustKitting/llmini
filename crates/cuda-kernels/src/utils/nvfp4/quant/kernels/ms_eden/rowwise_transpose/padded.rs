use cuda_device::{DisjointSlice, cuda_module, kernel, warp};

use super::super::HADAMARD_DIM;
use super::super::input::rowwise_transposed_hadamard_input;
use super::super::pack::{guarded_pack_chunk, ms_eden_pack_chunk};
use super::super::transpose_kernels::pack_padded_transpose_chunk;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(crate) mod module {
    use super::*;

    #[kernel]
    pub fn rowwise_nvfp4_transpose_to_nvfp4_ms_eden_device_scale_kernel(
        bytes: &[u8],
        scales: &[u8],
        source_global_scales: &[f32],
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
        pack_padded_transpose_chunk!(
            chunk_amax,
            input: rowwise_transposed_hadamard_input(bytes, scales, source_global_scales),
            chunk: chunk,
            output: [
                &mut out_fp4,
                &mut out_scales,
                core::ptr::null_mut(),
                &mut out_chunk_amax
            ],
            dims: [source_rows, source_cols, dst_row_len],
            scale: [global_scale[0], scale_override, scale_seed],
            sign_seed: sign_seed,
        );
    }
}

pub(crate) use module::{LoadedModule, from_module};
