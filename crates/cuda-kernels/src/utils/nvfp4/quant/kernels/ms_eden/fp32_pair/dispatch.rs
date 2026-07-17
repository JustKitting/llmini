macro_rules! dispatch_fp32_pair {
    (
        row_grid_dim: $row_grid_dim:expr,
        x: $x:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr],
        transpose_output: [$transpose_out_fp4:expr, $transpose_out_scales:expr, $transpose_out_global_scales:expr],
        scale: [
            $global_scale:expr, $scale_override:expr, $sign_seed:expr, $scale_seed:expr, $transpose_scale_seed:expr
        ],
        row: $row_body:ident($($row_arg:expr),* $(,)?);
        transpose: $transpose_body:ident($($transpose_arg:expr),* $(,)?)
    ) => {{
        let block = cuda_device::thread::blockIdx_x();
        let warp_in_block = cuda_device::thread::threadIdx_x() / 32;
        if block < $row_grid_dim {
            let chunk = block * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $row_body(
                $x,
                &mut $out_fp4,
                &mut $out_scales,
                &mut $out_global_scales,
                chunk,
                $($row_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $scale_seed,
            );
        } else {
            let chunk = (block - $row_grid_dim) * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $transpose_body(
                $x,
                &mut $transpose_out_fp4,
                &mut $transpose_out_scales,
                &mut $transpose_out_global_scales,
                chunk,
                $($transpose_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $transpose_scale_seed,
            );
        }
    }};
}

macro_rules! dispatch_fp32_pair_tiled {
    (
        row_grid_dim: $row_grid_dim:expr,
        source_rows: $source_rows:expr,
        source_cols: $source_cols:expr,
        transpose_chunks: $transpose_chunks:expr,
        transpose_chunks_are_shift: $transpose_chunks_are_shift:expr,
        x: $x:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr],
        transpose_output: [$transpose_out_fp4:expr, $transpose_out_scales:expr, $transpose_out_global_scales:expr],
        scale: [
            $global_scale:expr, $scale_override:expr, $sign_seed:expr, $scale_seed:expr, $transpose_scale_seed:expr
        ],
        row: $row_body:ident($($row_arg:expr),* $(,)?)
    ) => {{
        let block = cuda_device::thread::blockIdx_x();
        let warp_in_block = cuda_device::thread::threadIdx_x() / 32;
        if block < $row_grid_dim {
            let chunk = block * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $row_body(
                $x,
                &mut $out_fp4,
                &mut $out_scales,
                &mut $out_global_scales,
                chunk,
                $($row_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $scale_seed,
            );
        } else {
            static mut TILE: cuda_device::SharedArray<
                f32,
                { super::FP32_PAIR_TRANSPOSE_TILE_ELEMS },
            > = cuda_device::SharedArray::UNINIT;

            let source_rows = $source_rows;
            let source_cols = $source_cols;
            let transpose_block = block - $row_grid_dim;
            let source_row_tile_count =
                source_rows / super::FP32_PAIR_TRANSPOSE_TILE_ROWS as u32;
            let source_row_tile = transpose_block % source_row_tile_count;
            let source_col_tile = transpose_block / source_row_tile_count;
            let thread_id = cuda_device::thread::threadIdx_x() as usize;
            let tile_row = thread_id / super::FP32_PAIR_TRANSPOSE_TILE_COLS;
            let tile_col = thread_id
                - tile_row * super::FP32_PAIR_TRANSPOSE_TILE_COLS;
            let source_row_base =
                source_row_tile * super::FP32_PAIR_TRANSPOSE_TILE_ROWS as u32;
            let source_col_base =
                source_col_tile * super::FP32_PAIR_TRANSPOSE_TILE_COLS as u32;
            let source_row = source_row_base + tile_row as u32;
            let source_col = source_col_base + tile_col as u32;

            unsafe {
                TILE[tile_row * super::FP32_PAIR_TRANSPOSE_TILE_STRIDE + tile_col] =
                    crate::f16_tc_matmul::convert::load_f32_global_read_only(
                        $x.as_ptr(),
                        (source_row * source_cols + source_col) as usize,
                    );
            }
            cuda_device::thread::sync_threads();

            let lane = cuda_device::warp::lane_id();
            let row = source_col_base + warp_in_block;
            let chunks = $transpose_chunks;
            let chunk = if $transpose_chunks_are_shift {
                (row << chunks) + source_row_tile
            } else {
                row * chunks + source_row_tile
            };
            let input_col = source_row_base + lane;
            let input = unsafe {
                TILE[lane as usize * super::FP32_PAIR_TRANSPOSE_TILE_STRIDE
                    + warp_in_block as usize]
            } * random_sign($sign_seed, input_col);

            ms_eden_pack_chunk_no_chunk_amax_row(
                input,
                &mut $transpose_out_fp4,
                &mut $transpose_out_scales,
                &mut $transpose_out_global_scales,
                chunk,
                row,
                source_row_tile == 0,
                $global_scale[0],
                $scale_override,
                $transpose_scale_seed,
            );
        }
    }};
}

macro_rules! dispatch_fp32_pair_tiled_bias {
    (
        row_grid_dim: $row_grid_dim:expr,
        source_rows: $source_rows:expr,
        source_cols: $source_cols:expr,
        transpose_chunks: $transpose_chunks:expr,
        transpose_chunks_are_shift: $transpose_chunks_are_shift:expr,
        x: $x:expr,
        output: [$out_fp4:expr, $out_scales:expr, $out_global_scales:expr],
        transpose_output: [$transpose_out_fp4:expr, $transpose_out_scales:expr, $transpose_out_global_scales:expr],
        bias_output: $out_bias:expr,
        scale: [
            $global_scale:expr, $scale_override:expr, $sign_seed:expr, $scale_seed:expr, $transpose_scale_seed:expr
        ],
        row: $row_body:ident($($row_arg:expr),* $(,)?)
    ) => {{
        let block = cuda_device::thread::blockIdx_x();
        let warp_in_block = cuda_device::thread::threadIdx_x() / 32;
        if block < $row_grid_dim {
            let chunk = block * super::AMAX_WARPS_PER_BLOCK + warp_in_block;
            $row_body(
                $x,
                &mut $out_fp4,
                &mut $out_scales,
                &mut $out_global_scales,
                chunk,
                $($row_arg,)*
                $global_scale[0],
                $scale_override,
                $sign_seed,
                $scale_seed,
            );
        } else {
            static mut TILE: cuda_device::SharedArray<
                f32,
                { 2 * super::FP32_PAIR_TRANSPOSE_TILE_ELEMS },
            > = cuda_device::SharedArray::UNINIT;

            let source_rows = $source_rows;
            let source_cols = $source_cols;
            let source_col_tile = block - $row_grid_dim;
            let source_row_tile_count =
                source_rows / super::FP32_PAIR_TRANSPOSE_TILE_ROWS as u32;
            let source_col_base =
                source_col_tile * super::FP32_PAIR_TRANSPOSE_TILE_COLS as u32;
            let thread_id = cuda_device::thread::threadIdx_x() as usize;
            let tile_row = thread_id / super::FP32_PAIR_TRANSPOSE_TILE_COLS;
            let tile_col = thread_id
                - tile_row * super::FP32_PAIR_TRANSPOSE_TILE_COLS;
            let lane = cuda_device::warp::lane_id();
            let row = source_col_base + warp_in_block;
            let chunks = $transpose_chunks;
            let mut local_bias = 0.0f32;
            let source_col = source_col_base + tile_col as u32;

            let first_source_row = tile_row as u32;
            unsafe {
                TILE[tile_row * super::FP32_PAIR_TRANSPOSE_TILE_STRIDE + tile_col] =
                    crate::f16_tc_matmul::convert::load_f32_global_read_only(
                        $x.as_ptr(),
                        (first_source_row * source_cols + source_col) as usize,
                    );
            }
            cuda_device::thread::sync_threads();

            let mut source_row_tile = 0u32;
            while source_row_tile < source_row_tile_count {
                let tile_offset = (source_row_tile as usize & 1)
                    * super::FP32_PAIR_TRANSPOSE_TILE_ELEMS;
                let source_row_base =
                    source_row_tile * super::FP32_PAIR_TRANSPOSE_TILE_ROWS as u32;
                let raw_input = unsafe {
                    TILE[tile_offset
                        + lane as usize * super::FP32_PAIR_TRANSPOSE_TILE_STRIDE
                        + warp_in_block as usize]
                };

                let next_source_row_tile = source_row_tile + 1;
                if next_source_row_tile < source_row_tile_count {
                    let next_tile_offset = (next_source_row_tile as usize & 1)
                        * super::FP32_PAIR_TRANSPOSE_TILE_ELEMS;
                    let next_source_row = next_source_row_tile
                        * super::FP32_PAIR_TRANSPOSE_TILE_ROWS as u32
                        + tile_row as u32;
                    unsafe {
                        TILE[next_tile_offset
                            + tile_row * super::FP32_PAIR_TRANSPOSE_TILE_STRIDE
                            + tile_col] = crate::f16_tc_matmul::convert::load_f32_global_read_only(
                            $x.as_ptr(),
                            (next_source_row * source_cols + source_col) as usize,
                        );
                    }
                }

                local_bias += raw_input;
                let chunk = if $transpose_chunks_are_shift {
                    (row << chunks) + source_row_tile
                } else {
                    row * chunks + source_row_tile
                };
                let input_col = source_row_base + lane;
                let input = raw_input * random_sign($sign_seed, input_col);

                ms_eden_pack_chunk_no_chunk_amax_row(
                    input,
                    &mut $transpose_out_fp4,
                    &mut $transpose_out_scales,
                    &mut $transpose_out_global_scales,
                    chunk,
                    row,
                    source_row_tile == 0,
                    $global_scale[0],
                    $scale_override,
                    $transpose_scale_seed,
                );

                if next_source_row_tile < source_row_tile_count {
                    cuda_device::thread::sync_threads();
                }
                source_row_tile = next_source_row_tile;
            }

            let bias = warp_sum_f32(local_bias);
            if lane == 0 {
                unsafe {
                    *$out_bias.get_unchecked_mut(row as usize) = bias;
                }
            }
        }
    }};
}

pub(super) use {dispatch_fp32_pair, dispatch_fp32_pair_tiled, dispatch_fp32_pair_tiled_bias};
