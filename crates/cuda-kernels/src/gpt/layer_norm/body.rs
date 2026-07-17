macro_rules! maybe_store_residual_f16 {
    (none, $row_base:expr, $cols:expr, $embedding_dim:expr, $values:expr) => {};
    ($residual_f16:ident, $row_base:expr, $cols:expr, $embedding_dim:expr, $values:expr) => {
        $crate::layer_norm_utils::layer_norm_store_f16_3!(
            &mut $residual_f16,
            $row_base,
            $cols,
            $embedding_dim,
            $values
        );
    };
}

macro_rules! gpt_layer_norm_body {
    (
        $residual:ident $weight_bytes:ident $weight_scales:ident $bias_bytes:ident $bias_scales:ident;
        $weight_global_scale:ident $bias_global_scale:ident;
        $normalized:ident $normalized_amax:ident $mean_out:ident $inv_std_out:ident;
        $row_count:ident $embedding_dim:ident $epsilon:ident $output_scale:ident;
        $residual_f16:ident
    ) => {{
        use cuda_device::{SharedArray, thread};
        use $crate::float_ptx::sqrt_f32;
        use $crate::layer_norm::{
            GPT_LAYER_NORM_THREADS_PER_BLOCK, GPT_LAYER_NORM_WARPS_PER_BLOCK,
        };
        use $crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
        use $crate::layer_norm_utils::{
            centered_column, f32_column, layer_norm_columns3, layer_norm_map3,
            layer_norm_map3_indexed, layer_norm_square_sum3, layer_norm_store3, layer_norm_sum3,
            max_abs3, nvfp4_affine_normalized_column,
        };
        use $crate::warp_reduce::{thread_lane_warp, warp_max_f32, warp_sum_f32};

        static mut WARP_SUMS: SharedArray<f32, { GPT_LAYER_NORM_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let (thread, lane, warp_in_block) = thread_lane_warp();

        if row < $row_count {
            let row_base = row as usize * $embedding_dim as usize;
            let cols = layer_norm_columns3!(thread, GPT_LAYER_NORM_THREADS_PER_BLOCK);
            let values = layer_norm_map3!(cols, |col| f32_column(
                $residual,
                row_base,
                col,
                $embedding_dim
            ));

            maybe_store_residual_f16!($residual_f16, row_base, cols, $embedding_dim, values);

            let mean = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_sum3!(values),
                lane,
                warp_in_block,
                warp_sum_f32
            ) / $embedding_dim as f32;
            layer_norm_store_row!(&mut $mean_out, row, lane, warp_in_block, mean);
            let centered = layer_norm_map3_indexed!(cols, |index, col| centered_column(
                col,
                $embedding_dim,
                values[index],
                mean
            ));
            let variance_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                layer_norm_square_sum3!(centered),
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_std = 1.0 / sqrt_f32(variance_sum / $embedding_dim as f32 + $epsilon);
            layer_norm_store_row!(&mut $inv_std_out, row, lane, warp_in_block, inv_std);
            let affine_values =
                layer_norm_map3_indexed!(cols, |index, col| nvfp4_affine_normalized_column(
                    $weight_bytes,
                    $weight_scales,
                    $bias_bytes,
                    $bias_scales,
                    col,
                    $embedding_dim,
                    centered[index],
                    inv_std,
                    $weight_global_scale[0],
                    $bias_global_scale[0],
                ));
            let normalized_values = layer_norm_map3!(affine_values, |value| value * $output_scale);

            layer_norm_store3!(
                &mut $normalized,
                row_base,
                cols,
                $embedding_dim,
                normalized_values
            );

            let local_amax = max_abs3(
                normalized_values[0],
                normalized_values[1],
                normalized_values[2],
            );
            let block_amax = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                local_amax,
                lane,
                warp_in_block,
                warp_max_f32
            );

            layer_norm_store_row!(&mut $normalized_amax, row, lane, warp_in_block, block_amax);
        }
    }};
}

pub(super) use {gpt_layer_norm_body, maybe_store_residual_f16};
