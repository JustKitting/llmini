use cuda_device::{DisjointSlice, warp};

use super::super::HADAMARD_DIM;
use super::super::input::{hadamard_input, transposed_hadamard_input};
use super::super::pack::ms_eden_pack_chunk;

macro_rules! ms_eden_padded_body {
    ($name:ident, $input_fn:ident, [$($input_arg:ident),+], $dst_row_len:ident, chunk_amax) => {
        #[inline(always)]
        pub(in super::super) fn $name(
            x: &[f32],
            out_fp4: &mut DisjointSlice<'_, u8>,
            out_scales: &mut DisjointSlice<'_, u8>,
            out_global_scales: *mut f32,
            out_chunk_amax: &mut DisjointSlice<'_, f32>,
            chunk: u32,
            $($input_arg: u32,)+
            global_scale: f32,
            scale_override: f32,
            sign_seed: u32,
            scale_seed: u32,
        ) {
            let lane = warp::lane_id();
            let chunk_base = chunk * HADAMARD_DIM;
            let input = $input_fn(x, chunk_base, lane, $($input_arg,)+ sign_seed);

            ms_eden_pack_chunk(
                input,
                out_fp4,
                out_scales,
                out_global_scales,
                out_chunk_amax,
                chunk,
                $dst_row_len,
                global_scale,
                scale_override,
                scale_seed,
            );
        }
    };

}

ms_eden_padded_body!(
    fp32_to_nvfp4_ms_eden_body,
    hadamard_input,
    [src_row_len, dst_row_len],
    dst_row_len,
    chunk_amax
);
ms_eden_padded_body!(
    fp32_transpose_to_nvfp4_ms_eden_body,
    transposed_hadamard_input,
    [source_rows, dst_row_len, source_cols],
    dst_row_len,
    chunk_amax
);
