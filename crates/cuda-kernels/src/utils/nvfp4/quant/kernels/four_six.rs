use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::block_max_shared_f32_for_warps;
use crate::float_ptx::{abs_f32, max_f32, sqrt_f32};
use crate::warp_reduce::thread_lane_warp;

#[path = "four_six/helpers.rs"]
pub(crate) mod helpers;

const TRANSPOSE_TILE_ROWS: usize = 16;
const TRANSPOSE_TILE_COLS: usize = 64;
const TRANSPOSE_TILE_STRIDE: usize = TRANSPOSE_TILE_COLS + 1;
const TRANSPOSE_TILE_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_STRIDE;
const TRANSPOSE_TILE_LOAD_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_COLS;
const TRANSPOSE_GROUPS_PER_ROUND: usize = 256 / helpers::GROUP_THREADS;
const TRANSPOSE_TILE_GROUPS: usize = TRANSPOSE_TILE_LOAD_ELEMS / helpers::GROUP_SIZE;
const ROWWISE_WARPS_PER_BLOCK: usize = 8;
const ROWWISE_GROUPS_PER_BLOCK: u32 = 32;

#[cuda_module]
pub(crate) mod module {
    use super::helpers::*;
    use super::*;

    #[derive(Clone, Copy)]
    struct FourSixGroup {
        group: usize,
        base: usize,
        lane: usize,
        mask: u32,
        leader: u32,
    }

    struct FourSixOutputs<'a> {
        fp4: DisjointSlice<'a, u8>,
        scales: DisjointSlice<'a, u8>,
        global_scale: DisjointSlice<'a, f32>,
    }

    #[inline(always)]
    fn four_six_group_ctx() -> FourSixGroup {
        let (lane, mask, leader) = four_six_lane();
        let group = four_six_block_group();
        FourSixGroup {
            group,
            base: group * GROUP_SIZE,
            lane,
            mask,
            leader,
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_len: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let row_len = row_len as usize;
            let scalar_scale = row_len == 0;
            let scale_row_len = if scalar_scale { usize::MAX } else { row_len };
            let row = group_ctx.base / scale_row_len;
            let writes_global_scale = if scalar_scale {
                group_ctx.group == 0
            } else {
                group_ctx.base == row * scale_row_len
            };
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group(
                x,
                amax,
                out,
                group_ctx,
                row,
                writes_global_scale,
                scale_override,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_pow2_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_shift: u32,
        row_mask: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();
        let row = (group_ctx.base as u32 >> row_shift) as usize;
        let writes_global_scale = (group_ctx.base as u32 & row_mask) == 0;
        let out = FourSixOutputs {
            fp4: out_fp4,
            scales: out_scales,
            global_scale: out_global_scale,
        };
        pack_four_six_group(
            x,
            amax,
            out,
            group_ctx,
            row,
            writes_global_scale,
            scale_override,
        );
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_derived_amax_pow2_kernel(
        x: &[f32],
        mut amax: DisjointSlice<f32>,
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        row_len: u32,
        scale_override: f32,
    ) {
        static mut ROW_AMAX: SharedArray<f32, ROWWISE_WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        let (thread_id, lane, warp_in_block) = thread_lane_warp();
        if row >= row_count {
            return;
        }

        let row_base = row as usize * row_len as usize;
        let mut local_amax = 0.0_f32;
        let mut col = thread_id;
        while col < row_len {
            local_amax = max_f32(local_amax, abs_f32(x[row_base + col as usize]));
            col += thread::blockDim_x();
        }
        let tensor_amax = unsafe {
            block_max_shared_f32_for_warps(
                &mut ROW_AMAX,
                ROWWISE_WARPS_PER_BLOCK as u32,
                local_amax,
                lane,
                warp_in_block,
                0.0,
            )
        };
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        if thread_id == 0 {
            unsafe {
                *amax.get_unchecked_mut(row as usize) = tensor_amax;
                *out_global_scale.get_unchecked_mut(row as usize) = global_scale;
            }
        }

        let (lane_in_group, group_mask, group_leader) = four_six_lane();
        let groups_per_row = row_len / GROUP_SIZE as u32;
        let mut group_in_row = thread_id / GROUP_THREADS as u32;
        while group_in_row < groups_per_row {
            let group = row * groups_per_row + group_in_row;
            let base = group as usize * GROUP_SIZE;
            let value_lo = x[base + lane_in_group];
            let value_hi = x[base + lane_in_group + GROUP_THREADS];
            let (scale_bits, payload_pair) = four_six_group_scale(
                value_lo,
                value_hi,
                global_scale,
                scale_override,
                group_mask,
                group_leader,
                lane_in_group,
            );
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, group_mask);
            unsafe {
                if lane_in_group == 0 {
                    *out_scales.get_unchecked_mut(group as usize) = scale_bits;
                }
                if lane_in_group.is_multiple_of(2) {
                    *out_fp4.get_unchecked_mut(base / 2 + lane_in_group / 2) = payload_lo;
                    *out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane_in_group / 2) =
                        payload_hi;
                }
            }
            group_in_row += ROWWISE_GROUPS_PER_BLOCK;
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_padded_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value_lo = padded_value(x, base + group_ctx.lane as u32, rows, cols, padded_cols);
            let value_hi = padded_value(
                x,
                base + group_ctx.lane as u32 + GROUP_THREADS as u32,
                rows,
                cols,
                padded_cols,
            );
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value_lo,
                value_hi,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_exact_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group(
                x,
                amax,
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
            );
        }
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_four_six_exact_pow2_tiled_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales: DisjointSlice<u8>,
        mut transpose_out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        static mut TILE: SharedArray<f32, TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let source_row_base = thread::blockIdx_y() as usize * TRANSPOSE_TILE_ROWS;
        let source_col_base = thread::blockIdx_x() as usize * TRANSPOSE_TILE_COLS;
        let source_cols_usize = source_cols as usize;

        let mut load_offset = thread_id;
        while load_offset < TRANSPOSE_TILE_LOAD_ELEMS {
            let row = load_offset / TRANSPOSE_TILE_COLS;
            let col = load_offset - row * TRANSPOSE_TILE_COLS;
            unsafe {
                TILE[row * TRANSPOSE_TILE_STRIDE + col] =
                    x[(source_row_base + row) * source_cols_usize + source_col_base + col];
            }
            load_offset += 256;
        }
        thread::sync_threads();

        let (lane, mask, leader) = four_six_lane();
        let thread_group = thread_id / GROUP_THREADS;
        let global_scale = four_six_global_scale(amax[0], scale_override);
        if source_row_base == 0 && source_col_base == 0 && thread_id == 0 {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) = global_scale;
                *transpose_out_global_scale.get_unchecked_mut(0) = global_scale;
            }
        }

        let groups_per_source_row = source_cols_usize / GROUP_SIZE;
        let groups_per_tile_row = TRANSPOSE_TILE_COLS / GROUP_SIZE;
        let mut row_group = thread_group;
        while row_group < TRANSPOSE_TILE_GROUPS {
            let tile_row = row_group / groups_per_tile_row;
            let row_col = (row_group - tile_row * groups_per_tile_row) * GROUP_SIZE;
            let value_lo = unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + row_col + lane] };
            let value_hi =
                unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + row_col + lane + GROUP_THREADS] };
            let source_row = source_row_base + tile_row;
            let group =
                source_row * groups_per_source_row + (source_col_base + row_col) / GROUP_SIZE;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_pair) = four_six_group_scale(
                value_lo,
                value_hi,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, mask);

            unsafe {
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                if lane.is_multiple_of(2) {
                    *out_fp4.get_unchecked_mut(base / 2 + lane / 2) = payload_lo;
                    *out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane / 2) =
                        payload_hi;
                }
            }
            row_group += TRANSPOSE_GROUPS_PER_ROUND;
        }

        let groups_per_output_row = source_rows as usize / GROUP_SIZE;
        let group_in_output_row = source_row_base / GROUP_SIZE;
        let mut transpose_col = thread_group;
        while transpose_col < TRANSPOSE_TILE_COLS {
            let value_lo = unsafe { TILE[lane * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let value_hi =
                unsafe { TILE[(lane + GROUP_THREADS) * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let group =
                (source_col_base + transpose_col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_pair) = four_six_group_scale(
                value_lo,
                value_hi,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, mask);

            unsafe {
                if lane == 0 {
                    *transpose_out_scales.get_unchecked_mut(group) = scale_bits;
                }
                if lane.is_multiple_of(2) {
                    *transpose_out_fp4.get_unchecked_mut(base / 2 + lane / 2) = payload_lo;
                    *transpose_out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane / 2) =
                        payload_hi;
                }
            }
            transpose_col += TRANSPOSE_GROUPS_PER_ROUND;
        }
    }

    #[kernel]
    pub fn four_six_rebase_sqrt_bound_global_scale_kernel(
        original_amax: &[f32],
        sqrt_bound_amax: &[f32],
        mut out_global_scale: DisjointSlice<f32>,
    ) {
        if thread::threadIdx_x() == 0 {
            let bound = sqrt_bound_amax[0];
            let scale = if bound > 1.0 {
                1.0 / sqrt_f32(bound)
            } else {
                1.0
            };
            unsafe {
                *out_global_scale.get_unchecked_mut(0) =
                    four_six_global_scale(original_amax[0] * scale, 1.0);
            }
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_exact_bounded_amax_kernel(
        x: &[f32],
        original_amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        apply_bound: u32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let bound = original_amax[0];
            let bounded_amax = if bound > 1.0 {
                bound * (1.0 / bound)
            } else {
                bound
            };
            let bound_scale = if apply_bound != 0 && bound > 1.0 {
                1.0 / bound
            } else {
                1.0
            };
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                bounded_amax,
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                1.0,
                x[group_ctx.base + group_ctx.lane] * bound_scale,
                x[group_ctx.base + group_ctx.lane + GROUP_THREADS] * bound_scale,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_padded_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value_lo = transposed_padded_value(
                x,
                base + group_ctx.lane as u32,
                source_rows,
                source_cols,
                padded_cols,
            );
            let value_hi = transposed_padded_value(
                x,
                base + group_ctx.lane as u32 + GROUP_THREADS as u32,
                source_rows,
                source_cols,
                padded_cols,
            );
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value_lo,
                value_hi,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value_lo =
                transposed_exact_value(x, base + group_ctx.lane as u32, source_rows, source_cols);
            let value_hi = transposed_exact_value(
                x,
                base + group_ctx.lane as u32 + GROUP_THREADS as u32,
                source_rows,
                source_cols,
            );
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value_lo,
                value_hi,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_pow2_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        source_rows_shift: u32,
        source_rows_mask: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        let group_ctx = four_six_group_ctx();

        if group_ctx.group < out_scales.len() {
            let base = group_ctx.base as u32;
            let value_lo = transposed_exact_value_pow2(
                x,
                base + group_ctx.lane as u32,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let value_hi = transposed_exact_value_pow2(
                x,
                base + group_ctx.lane as u32 + GROUP_THREADS as u32,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let out = FourSixOutputs {
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            pack_four_six_group_values(
                amax[0],
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                scale_override,
                value_lo,
                value_hi,
            );
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        scale_override: f32,
    ) {
        static mut TILE: SharedArray<f32, TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let source_row_base = thread::blockIdx_y() as usize * TRANSPOSE_TILE_ROWS;
        let source_col_base = thread::blockIdx_x() as usize * TRANSPOSE_TILE_COLS;
        let source_cols_usize = source_cols as usize;

        let mut load_offset = thread_id;
        while load_offset < TRANSPOSE_TILE_LOAD_ELEMS {
            let row = load_offset / TRANSPOSE_TILE_COLS;
            let col = load_offset - row * TRANSPOSE_TILE_COLS;
            unsafe {
                TILE[row * TRANSPOSE_TILE_STRIDE + col] =
                    x[(source_row_base + row) * source_cols_usize + source_col_base + col];
            }
            load_offset += 256;
        }
        thread::sync_threads();

        let (lane, mask, leader) = four_six_lane();
        let thread_group = thread_id / GROUP_THREADS;
        let groups_per_output_row = source_rows as usize / GROUP_SIZE;
        let group_in_output_row = source_row_base / GROUP_SIZE;
        let global_scale = four_six_global_scale(amax[0], scale_override);
        let mut col = thread_group;

        while col < TRANSPOSE_TILE_COLS {
            let value_lo = unsafe { TILE[lane * TRANSPOSE_TILE_STRIDE + col] };
            let value_hi = unsafe { TILE[(lane + GROUP_THREADS) * TRANSPOSE_TILE_STRIDE + col] };
            let group = (source_col_base + col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_pair) = four_six_group_scale(
                value_lo,
                value_hi,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, mask);

            unsafe {
                if group == 0 && lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                if lane.is_multiple_of(2) {
                    *out_fp4.get_unchecked_mut(base / 2 + lane / 2) = payload_lo;
                    *out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane / 2) =
                        payload_hi;
                }
            }

            col += TRANSPOSE_GROUPS_PER_ROUND;
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_sqrt_bounded_amax_kernel(
        x: &[f32],
        original_amax: &[f32],
        sqrt_bound_amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        apply_bound: u32,
    ) {
        static mut TILE: SharedArray<f32, TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let source_row_base = thread::blockIdx_y() as usize * TRANSPOSE_TILE_ROWS;
        let source_col_base = thread::blockIdx_x() as usize * TRANSPOSE_TILE_COLS;
        let source_cols_usize = source_cols as usize;

        let mut load_offset = thread_id;
        while load_offset < TRANSPOSE_TILE_LOAD_ELEMS {
            let row = load_offset / TRANSPOSE_TILE_COLS;
            let col = load_offset - row * TRANSPOSE_TILE_COLS;
            unsafe {
                TILE[row * TRANSPOSE_TILE_STRIDE + col] =
                    x[(source_row_base + row) * source_cols_usize + source_col_base + col];
            }
            load_offset += 256;
        }
        thread::sync_threads();

        let (lane, mask, leader) = four_six_lane();
        let thread_group = thread_id / GROUP_THREADS;
        let groups_per_output_row = source_rows as usize / GROUP_SIZE;
        let group_in_output_row = source_row_base / GROUP_SIZE;
        let bound = sqrt_bound_amax[0];
        let scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let global_scale = four_six_global_scale(original_amax[0] * scale, 1.0);
        let mut col = thread_group;

        while col < TRANSPOSE_TILE_COLS {
            let mut value_lo = unsafe { TILE[lane * TRANSPOSE_TILE_STRIDE + col] };
            let mut value_hi =
                unsafe { TILE[(lane + GROUP_THREADS) * TRANSPOSE_TILE_STRIDE + col] };
            if apply_bound != 0 {
                value_lo *= scale;
                value_hi *= scale;
            }
            let group = (source_col_base + col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_pair) =
                four_six_group_scale(value_lo, value_hi, global_scale, 1.0, mask, leader, lane);
            let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, mask);

            unsafe {
                if group == 0 && lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                if lane.is_multiple_of(2) {
                    *out_fp4.get_unchecked_mut(base / 2 + lane / 2) = payload_lo;
                    *out_fp4.get_unchecked_mut(base / 2 + GROUP_THREADS / 2 + lane / 2) =
                        payload_hi;
                }
            }

            col += TRANSPOSE_GROUPS_PER_ROUND;
        }
    }

    fn pack_four_six_group(
        x: &[f32],
        amax: &[f32],
        out: FourSixOutputs<'_>,
        group_ctx: FourSixGroup,
        row: usize,
        writes_global_scale: bool,
        scale_override: f32,
    ) {
        let value_lo = x[group_ctx.base + group_ctx.lane];
        let value_hi = x[group_ctx.base + group_ctx.lane + GROUP_THREADS];
        pack_four_six_group_values(
            amax[row],
            out,
            group_ctx,
            row,
            writes_global_scale,
            scale_override,
            value_lo,
            value_hi,
        );
    }

    #[expect(clippy::too_many_arguments, reason = "device pack path is explicit")]
    fn pack_four_six_group_values(
        tensor_amax: f32,
        mut out: FourSixOutputs<'_>,
        group_ctx: FourSixGroup,
        scale_row: usize,
        writes_global_scale: bool,
        scale_override: f32,
        value_lo: f32,
        value_hi: f32,
    ) {
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        let (scale_bits, payload_pair) = four_six_group_scale(
            value_lo,
            value_hi,
            global_scale,
            scale_override,
            group_ctx.mask,
            group_ctx.leader,
            group_ctx.lane,
        );
        let (payload_lo, payload_hi) = four_six_payload_bytes(payload_pair, group_ctx.mask);

        unsafe {
            if writes_global_scale && group_ctx.lane == 0 {
                *out.global_scale.get_unchecked_mut(scale_row) = global_scale;
            }
            if group_ctx.lane == 0 {
                *out.scales.get_unchecked_mut(group_ctx.group) = scale_bits;
            }
            if group_ctx.lane.is_multiple_of(2) {
                *out.fp4
                    .get_unchecked_mut(group_ctx.base / 2 + group_ctx.lane / 2) = payload_lo;
                *out.fp4.get_unchecked_mut(
                    group_ctx.base / 2 + GROUP_THREADS / 2 + group_ctx.lane / 2,
                ) = payload_hi;
            }
        }
    }

    fn padded_value(x: &[f32], output_index: u32, rows: u32, cols: u32, padded_cols: u32) -> f32 {
        let row = output_index / padded_cols;
        let col = output_index - row * padded_cols;
        if row < rows && col < cols {
            x[(row * cols + col) as usize]
        } else {
            0.0
        }
    }

    fn transposed_padded_value(
        x: &[f32],
        output_index: u32,
        source_rows: u32,
        source_cols: u32,
        padded_cols: u32,
    ) -> f32 {
        let row = output_index / padded_cols;
        let col = output_index - row * padded_cols;
        if row < source_cols && col < source_rows {
            x[(col * source_cols + row) as usize]
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn transposed_exact_value(
        x: &[f32],
        output_index: u32,
        source_rows: u32,
        source_cols: u32,
    ) -> f32 {
        let row = output_index / source_rows;
        let col = output_index - row * source_rows;
        x[(col * source_cols + row) as usize]
    }

    #[inline(always)]
    fn transposed_exact_value_pow2(
        x: &[f32],
        output_index: u32,
        source_rows_shift: u32,
        source_rows_mask: u32,
        source_cols: u32,
    ) -> f32 {
        let row = output_index >> source_rows_shift;
        let col = output_index & source_rows_mask;
        x[(col * source_cols + row) as usize]
    }
}
