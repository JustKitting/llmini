macro_rules! maybe_store_residual_f16 {
    (none, $row_base:expr, $col:expr, $embedding_dim:expr, $value:expr) => {};
    ($residual_f16:ident, $row_base:expr, $col:expr, $embedding_dim:expr, $value:expr) => {
        $crate::layer_norm_utils::store_f16_column(
            &mut $residual_f16,
            $row_base,
            $col,
            $embedding_dim,
            $value,
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
        use $crate::float_ptx::{abs_f32, max_f32, sqrt_f32};
        use $crate::layer_norm::{
            GPT_LAYER_NORM_THREADS_PER_BLOCK, GPT_LAYER_NORM_WARPS_PER_BLOCK,
        };
        use $crate::layer_norm_reduce::{layer_norm_block_reduce, layer_norm_store_row};
        use $crate::layer_norm_utils::nvfp4_affine_normalized_column;
        use $crate::warp_reduce::{thread_lane_warp, warp_max_f32, warp_sum_f32};

        static mut WARP_SUMS: SharedArray<f32, { GPT_LAYER_NORM_WARPS_PER_BLOCK as usize }> =
            SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let (thread, lane, warp_in_block) = thread_lane_warp();

        if row < $row_count {
            let row_base = row as usize * $embedding_dim as usize;
            let mut sum_local = 0.0f32;
            let mut col = thread;
            while col < $embedding_dim {
                let value = $residual[row_base + col as usize];
                sum_local += value;
                maybe_store_residual_f16!($residual_f16, row_base, col, $embedding_dim, value);
                col += GPT_LAYER_NORM_THREADS_PER_BLOCK;
            }
            let mean = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                sum_local,
                lane,
                warp_in_block,
                warp_sum_f32
            ) / $embedding_dim as f32;
            layer_norm_store_row!(&mut $mean_out, row, lane, warp_in_block, mean);
            let mut variance_local = 0.0f32;
            let mut col = thread;
            while col < $embedding_dim {
                let centered = $residual[row_base + col as usize] - mean;
                variance_local += centered * centered;
                col += GPT_LAYER_NORM_THREADS_PER_BLOCK;
            }
            let variance_sum = layer_norm_block_reduce!(
                WARP_SUMS,
                GPT_LAYER_NORM_WARPS_PER_BLOCK,
                variance_local,
                lane,
                warp_in_block,
                warp_sum_f32
            );
            let inv_std = 1.0 / sqrt_f32(variance_sum / $embedding_dim as f32 + $epsilon);
            layer_norm_store_row!(&mut $inv_std_out, row, lane, warp_in_block, inv_std);
            let mut local_amax = 0.0f32;
            let mut col = thread;
            while col < $embedding_dim {
                let centered = $residual[row_base + col as usize] - mean;
                let normalized = nvfp4_affine_normalized_column(
                    $weight_bytes,
                    $weight_scales,
                    $bias_bytes,
                    $bias_scales,
                    col,
                    $embedding_dim,
                    centered,
                    inv_std,
                    $weight_global_scale[0],
                    $bias_global_scale[0],
                ) * $output_scale;
                unsafe {
                    *$normalized.get_unchecked_mut(row_base + col as usize) = normalized;
                }
                local_amax = max_f32(local_amax, abs_f32(normalized));
                col += GPT_LAYER_NORM_THREADS_PER_BLOCK;
            }
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
