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

macro_rules! maybe_store_residual_f16_quad {
    (none,$base:expr,$x:expr) => {};
    ($out:ident,$base:expr,$x:expr) => {
        $crate::layer_norm_utils::store_f16_quad(&mut $out, $base, $x);
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
            if $embedding_dim == 2048 {
                use $crate::layer_norm_utils::{load_f32_quad, store_f32_quad};
                let base0 = row_base + thread as usize * 4;
                let base1 = base0 + 1024;
                let values0 = load_f32_quad($residual, base0);
                let values1 = load_f32_quad($residual, base1);
                maybe_store_residual_f16_quad!($residual_f16, base0, values0);
                maybe_store_residual_f16_quad!($residual_f16, base1, values1);
                let mut sum_local = 0.0f32;
                let value_0 = values0[0];
                sum_local += value_0;
                let value_1 = values0[1];
                sum_local += value_1;
                let value_2 = values0[2];
                sum_local += value_2;
                let value_3 = values0[3];
                sum_local += value_3;
                let value_4 = values1[0];
                sum_local += value_4;
                let value_5 = values1[1];
                sum_local += value_5;
                let value_6 = values1[2];
                sum_local += value_6;
                let value_7 = values1[3];
                sum_local += value_7;
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
                let centered_0 = value_0 - mean;
                variance_local += centered_0 * centered_0;
                let centered_1 = value_1 - mean;
                variance_local += centered_1 * centered_1;
                let centered_2 = value_2 - mean;
                variance_local += centered_2 * centered_2;
                let centered_3 = value_3 - mean;
                variance_local += centered_3 * centered_3;
                let centered_4 = value_4 - mean;
                variance_local += centered_4 * centered_4;
                let centered_5 = value_5 - mean;
                variance_local += centered_5 * centered_5;
                let centered_6 = value_6 - mean;
                variance_local += centered_6 * centered_6;
                let centered_7 = value_7 - mean;
                variance_local += centered_7 * centered_7;
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
                let weight0 = $crate::nvfp4::nvfp4_vector_values4(
                    $weight_bytes,
                    $weight_scales,
                    $weight_global_scale[0],
                    thread as usize * 4 + 0,
                );
                let bias0 = $crate::nvfp4::nvfp4_vector_values4(
                    $bias_bytes,
                    $bias_scales,
                    $bias_global_scale[0],
                    thread as usize * 4 + 0,
                );
                let n0 = $crate::float_ptx::fma_f32(centered_0 * inv_std, weight0[0], bias0[0])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n0));
                let n1 = $crate::float_ptx::fma_f32(centered_1 * inv_std, weight0[1], bias0[1])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n1));
                let n2 = $crate::float_ptx::fma_f32(centered_2 * inv_std, weight0[2], bias0[2])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n2));
                let n3 = $crate::float_ptx::fma_f32(centered_3 * inv_std, weight0[3], bias0[3])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n3));
                store_f32_quad(&mut $normalized, base0, [n0, n1, n2, n3]);
                let weight1 = $crate::nvfp4::nvfp4_vector_values4(
                    $weight_bytes,
                    $weight_scales,
                    $weight_global_scale[0],
                    thread as usize * 4 + 1024,
                );
                let bias1 = $crate::nvfp4::nvfp4_vector_values4(
                    $bias_bytes,
                    $bias_scales,
                    $bias_global_scale[0],
                    thread as usize * 4 + 1024,
                );
                let n4 = $crate::float_ptx::fma_f32(centered_4 * inv_std, weight1[0], bias1[0])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n4));
                let n5 = $crate::float_ptx::fma_f32(centered_5 * inv_std, weight1[1], bias1[1])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n5));
                let n6 = $crate::float_ptx::fma_f32(centered_6 * inv_std, weight1[2], bias1[2])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n6));
                let n7 = $crate::float_ptx::fma_f32(centered_7 * inv_std, weight1[3], bias1[3])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n7));
                store_f32_quad(&mut $normalized, base1, [n4, n5, n6, n7]);
                let block_amax = layer_norm_block_reduce!(
                    WARP_SUMS,
                    GPT_LAYER_NORM_WARPS_PER_BLOCK,
                    local_amax,
                    lane,
                    warp_in_block,
                    warp_max_f32
                );
                layer_norm_store_row!(&mut $normalized_amax, row, lane, warp_in_block, block_amax);
            } else {
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
        }
    }};
}

pub(super) use {gpt_layer_norm_body, maybe_store_residual_f16, maybe_store_residual_f16_quad};

macro_rules! gpt_layer_norm_xhat_body {
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
            if $embedding_dim == 2048 {
                use $crate::layer_norm_utils::{load_f32_quad, store_f32_quad};
                let base0 = row_base + thread as usize * 4;
                let base1 = base0 + 1024;
                let values0 = load_f32_quad($residual, base0);
                let values1 = load_f32_quad($residual, base1);
                let mut sum_local = 0.0f32;
                let value_0 = values0[0];
                sum_local += value_0;
                let value_1 = values0[1];
                sum_local += value_1;
                let value_2 = values0[2];
                sum_local += value_2;
                let value_3 = values0[3];
                sum_local += value_3;
                let value_4 = values1[0];
                sum_local += value_4;
                let value_5 = values1[1];
                sum_local += value_5;
                let value_6 = values1[2];
                sum_local += value_6;
                let value_7 = values1[3];
                sum_local += value_7;
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
                let centered_0 = value_0 - mean;
                variance_local += centered_0 * centered_0;
                let centered_1 = value_1 - mean;
                variance_local += centered_1 * centered_1;
                let centered_2 = value_2 - mean;
                variance_local += centered_2 * centered_2;
                let centered_3 = value_3 - mean;
                variance_local += centered_3 * centered_3;
                let centered_4 = value_4 - mean;
                variance_local += centered_4 * centered_4;
                let centered_5 = value_5 - mean;
                variance_local += centered_5 * centered_5;
                let centered_6 = value_6 - mean;
                variance_local += centered_6 * centered_6;
                let centered_7 = value_7 - mean;
                variance_local += centered_7 * centered_7;
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
                // Standardized coordinates are bounded by sqrt(width). Keep
                // their16-bit tape finite even when the raw residual exceeds65504.
                maybe_store_residual_f16_quad!(
                    $residual_f16,
                    base0,
                    [
                        centered_0 * inv_std,
                        centered_1 * inv_std,
                        centered_2 * inv_std,
                        centered_3 * inv_std
                    ]
                );
                maybe_store_residual_f16_quad!(
                    $residual_f16,
                    base1,
                    [
                        centered_4 * inv_std,
                        centered_5 * inv_std,
                        centered_6 * inv_std,
                        centered_7 * inv_std
                    ]
                );
                let mut local_amax = 0.0f32;
                let weight0 = $crate::nvfp4::nvfp4_vector_values4(
                    $weight_bytes,
                    $weight_scales,
                    $weight_global_scale[0],
                    thread as usize * 4 + 0,
                );
                let bias0 = $crate::nvfp4::nvfp4_vector_values4(
                    $bias_bytes,
                    $bias_scales,
                    $bias_global_scale[0],
                    thread as usize * 4 + 0,
                );
                let n0 = $crate::float_ptx::fma_f32(centered_0 * inv_std, weight0[0], bias0[0])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n0));
                let n1 = $crate::float_ptx::fma_f32(centered_1 * inv_std, weight0[1], bias0[1])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n1));
                let n2 = $crate::float_ptx::fma_f32(centered_2 * inv_std, weight0[2], bias0[2])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n2));
                let n3 = $crate::float_ptx::fma_f32(centered_3 * inv_std, weight0[3], bias0[3])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n3));
                store_f32_quad(&mut $normalized, base0, [n0, n1, n2, n3]);
                let weight1 = $crate::nvfp4::nvfp4_vector_values4(
                    $weight_bytes,
                    $weight_scales,
                    $weight_global_scale[0],
                    thread as usize * 4 + 1024,
                );
                let bias1 = $crate::nvfp4::nvfp4_vector_values4(
                    $bias_bytes,
                    $bias_scales,
                    $bias_global_scale[0],
                    thread as usize * 4 + 1024,
                );
                let n4 = $crate::float_ptx::fma_f32(centered_4 * inv_std, weight1[0], bias1[0])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n4));
                let n5 = $crate::float_ptx::fma_f32(centered_5 * inv_std, weight1[1], bias1[1])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n5));
                let n6 = $crate::float_ptx::fma_f32(centered_6 * inv_std, weight1[2], bias1[2])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n6));
                let n7 = $crate::float_ptx::fma_f32(centered_7 * inv_std, weight1[3], bias1[3])
                    * $output_scale;
                local_amax = max_f32(local_amax, abs_f32(n7));
                store_f32_quad(&mut $normalized, base1, [n4, n5, n6, n7]);
                let block_amax = layer_norm_block_reduce!(
                    WARP_SUMS,
                    GPT_LAYER_NORM_WARPS_PER_BLOCK,
                    local_amax,
                    lane,
                    warp_in_block,
                    warp_max_f32
                );
                layer_norm_store_row!(&mut $normalized_amax, row, lane, warp_in_block, block_amax);
            } else {
                let mut sum_local = 0.0f32;
                let mut col = thread;
                while col < $embedding_dim {
                    let value = $residual[row_base + col as usize];
                    sum_local += value;
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
                    maybe_store_residual_f16!(
                        $residual_f16,
                        row_base,
                        col,
                        $embedding_dim,
                        centered * inv_std
                    );
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
        }
    }};
}
pub(super) use gpt_layer_norm_xhat_body;
