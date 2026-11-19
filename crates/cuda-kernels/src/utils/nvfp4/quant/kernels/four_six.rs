use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

use crate::block_reduce::block_max_shared_f32_for_warps;
use crate::f16_tc_matmul::convert::load_f32x2_global;
use crate::float_ptx::{abs_f32, max_f32, sqrt_f32};
use crate::nvfp4_tma_matmul::cute::Sm120ScaleLayout;
use crate::warp_reduce::thread_lane_warp;

#[path = "four_six/helpers.rs"]
pub(crate) mod helpers;

const TRANSPOSE_TILE_ROWS: usize = 32;
const TRANSPOSE_TILE_COLS: usize = 64;
const TRANSPOSE_TILE_STRIDE: usize = TRANSPOSE_TILE_COLS + 1;
const TRANSPOSE_TILE_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_STRIDE;
const TRANSPOSE_TILE_LOAD_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_COLS;
const TRANSPOSE_GROUPS_PER_ROUND: usize = 256 / helpers::GROUP_THREADS;
const TRANSPOSE_TILE_GROUPS: usize = TRANSPOSE_TILE_LOAD_ELEMS / helpers::GROUP_SIZE;
const PACKED_TRANSPOSE_TILE_ELEMS: usize = TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_COLS;
const BOUNDED_TRANSPOSE_TILE_ROWS: usize = 16;
const BOUNDED_TRANSPOSE_TILE_STRIDE: usize = TRANSPOSE_TILE_COLS + 1;
const BOUNDED_TRANSPOSE_TILE_ELEMS: usize =
    BOUNDED_TRANSPOSE_TILE_ROWS * BOUNDED_TRANSPOSE_TILE_STRIDE;
const BOUNDED_TRANSPOSE_TILE_LOAD_ELEMS: usize = BOUNDED_TRANSPOSE_TILE_ROWS * TRANSPOSE_TILE_COLS;
const ROWWISE_WARPS_PER_BLOCK: usize = 8;
const ROWWISE_GROUPS_PER_BLOCK: u32 = 256 / helpers::GROUP_THREADS as u32;

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
        layout: u8,
        row_len: u32,
        row_count: u32,
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

    #[inline(always)]
    fn tiled_row_group_without_bank_conflicts(linear_group: usize) -> usize {
        let tile_row_chunk = linear_group / TRANSPOSE_GROUPS_PER_ROUND;
        let local_group = linear_group & (TRANSPOSE_GROUPS_PER_ROUND - 1);
        let warp = local_group / 8;
        let group_in_warp = local_group & 7;
        let tile_row = tile_row_chunk * 16 + (warp / 2) * 4 + group_in_warp / 2;
        let group_in_row = (warp & 1) * 2 + (group_in_warp & 1);
        tile_row * (TRANSPOSE_TILE_COLS / GROUP_SIZE) + group_in_row
    }

    #[inline(always)]
    fn tiled_transpose_group_without_bank_conflicts(linear_group: usize) -> usize {
        let col_half = linear_group / TRANSPOSE_GROUPS_PER_ROUND;
        let local_group = linear_group & (TRANSPOSE_GROUPS_PER_ROUND - 1);
        let warp = local_group / 8;
        let group_in_warp = local_group & 7;
        let tile_row_group = group_in_warp & 1;
        let col = col_half * 32 + warp * 4 + group_in_warp / 2;
        tile_row_group * TRANSPOSE_TILE_COLS + col
    }

    #[inline(always)]
    fn packed_tiled_row_group(linear_group: usize) -> usize {
        let tile_row_chunk = linear_group / TRANSPOSE_GROUPS_PER_ROUND;
        let local_group = linear_group & (TRANSPOSE_GROUPS_PER_ROUND - 1);
        let warp = local_group / 8;
        let group_in_warp = local_group & 7;
        let tile_row = tile_row_chunk * 16 + warp * 2 + group_in_warp / 4;
        let group_in_row = group_in_warp & 3;
        tile_row * (TRANSPOSE_TILE_COLS / GROUP_SIZE) + group_in_row
    }

    #[inline(always)]
    fn packed_transpose_tile_index(row: usize, col: usize) -> usize {
        let logical = row * TRANSPOSE_TILE_COLS + col;
        logical ^ ((col & 32) >> 4) ^ (row & 31)
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_kernel(
        x: &[f32],
        amax: &[f32],
        out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_len: u32,
        row_count: u32,
        scale_override: f32,
    ) {
        let mut group_ctx = four_six_group_ctx();
        if row_len != 0 {
            let local = thread::blockIdx_x() * 64 + thread::threadIdx_x() / 4;
            if local >= row_len / 16 {
                return;
            }
            group_ctx.group = (thread::blockIdx_y() * (row_len / 16) + local) as usize;
            group_ctx.base = group_ctx.group * GROUP_SIZE;
        }

        if group_ctx.group < out_fp4.len() / 8 {
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
                layout: if scalar_scale { 2 } else { 1 },
                row_len: row_len as u32,
                row_count,
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
        row_count: u32,
        scale_override: f32,
    ) {
        let mut group_ctx = four_six_group_ctx();
        let local = thread::blockIdx_x() * 64 + thread::threadIdx_x() / 4;
        if local >= (row_mask + 1) / 16 {
            return;
        }
        group_ctx.group = (thread::blockIdx_y() * ((row_mask + 1) / 16) + local) as usize;
        group_ctx.base = group_ctx.group * GROUP_SIZE;
        let row = (group_ctx.base as u32 >> row_shift) as usize;
        let writes_global_scale = (group_ctx.base as u32 & row_mask) == 0;
        let out = FourSixOutputs {
            layout: 1,
            row_len: row_mask + 1,
            row_count,
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
            let value_base = base + 4 * lane_in_group;
            let (value_0, value_1) = load_f32x2_global(x.as_ptr(), value_base);
            let (value_2, value_3) = load_f32x2_global(x.as_ptr(), value_base + 2);
            let (scale_bits, payload_word) = activation_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                group_mask,
                group_leader,
                lane_in_group,
            );
            unsafe {
                crate::nvfp4::csf::store_row_warp(
                    out_scales.as_mut_ptr(),
                    row_count,
                    row_len.div_ceil(128) * 128,
                    row,
                    group_in_row,
                    scale_bits,
                );
                store_four_six_payload_word(
                    out_fp4.as_mut_ptr(),
                    base,
                    lane_in_group,
                    payload_word,
                );
            }
            group_in_row += ROWWISE_GROUPS_PER_BLOCK;
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_derived_amax_2048_cached_kernel(
        x: &[f32],
        amax: DisjointSlice<f32>,
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        scale_override: f32,
    ) {
        static mut ROW_AMAX: SharedArray<f32, ROWWISE_WARPS_PER_BLOCK> = SharedArray::UNINIT;
        unsafe {
            pack_cached_rows::<2048, 2>(
                |row, column| {
                    let base = row as usize * 2048 + column;
                    let (a, b) = load_f32x2_global(x.as_ptr(), base);
                    let (c, d) = load_f32x2_global(x.as_ptr(), base + 2);
                    [a, b, c, d]
                },
                Some(amax),
                out_fp4,
                out_scales,
                out_global_scale,
                row_count,
                scale_override,
                &mut ROW_AMAX,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_rowwise_derived_amax_8192_cached_kernel(
        x: &[f32],
        amax: DisjointSlice<f32>,
        out_fp4: DisjointSlice<u8>,
        out_scales: DisjointSlice<u8>,
        out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        scale_override: f32,
    ) {
        static mut ROW_AMAX: SharedArray<f32, ROWWISE_WARPS_PER_BLOCK> = SharedArray::UNINIT;
        unsafe {
            rowwise_derived_amax_8192_cached_body::<false>(
                x,
                amax,
                out_fp4,
                out_scales,
                out_global_scale,
                row_count,
                scale_override,
                1.0,
                &mut ROW_AMAX,
            );
        }
    }

    #[derive(Clone, Copy)]
    struct CachedFourSixValues {
        value_0: f32,
        value_1: f32,
        value_2: f32,
        value_3: f32,
    }

    impl CachedFourSixValues {
        #[inline(always)]
        fn amax(self) -> f32 {
            let mut value = max_f32(abs_f32(self.value_0), abs_f32(self.value_1));
            value = max_f32(value, abs_f32(self.value_2));
            max_f32(value, abs_f32(self.value_3))
        }
    }

    #[inline(always)]
    fn load_8192_cached_values<const NORMALIZE: bool>(
        x: &[f32],
        row: u32,
        group_in_row: u32,
        lane_in_group: usize,
        inverse: f32,
    ) -> CachedFourSixValues {
        let base = row as usize * 8192 + group_in_row as usize * GROUP_SIZE + 4 * lane_in_group;
        let (value_0, value_1) = load_f32x2_global(x.as_ptr(), base);
        let (value_2, value_3) = load_f32x2_global(x.as_ptr(), base + 2);
        CachedFourSixValues {
            value_0: if NORMALIZE {
                value_0 * inverse
            } else {
                value_0
            },
            value_1: if NORMALIZE {
                value_1 * inverse
            } else {
                value_1
            },
            value_2: if NORMALIZE {
                value_2 * inverse
            } else {
                value_2
            },
            value_3: if NORMALIZE {
                value_3 * inverse
            } else {
                value_3
            },
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn store_8192_cached_values(
        values: CachedFourSixValues,
        out_fp4: &mut DisjointSlice<u8>,
        out_scales: &mut DisjointSlice<u8>,
        row: u32,
        group_in_row: u32,
        global_scale: f32,
        scale_override: f32,
        group_mask: u32,
        group_leader: u32,
        lane_in_group: usize,
        row_count: u32,
    ) {
        let group = row * (8192 / GROUP_SIZE as u32) + group_in_row;
        let base = group as usize * GROUP_SIZE;
        let (scale_bits, payload_word) = activation_grid_group_scale(
            values.value_0,
            values.value_1,
            values.value_2,
            values.value_3,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
        );
        unsafe {
            crate::nvfp4::csf::store_row_warp(
                out_scales.as_mut_ptr(),
                row_count,
                8192,
                row,
                group_in_row,
                scale_bits,
            );
            store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane_in_group, payload_word);
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn rowwise_derived_amax_8192_cached_body<const NORMALIZE: bool>(
        x: &[f32],
        mut amax: DisjointSlice<f32>,
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        scale_override: f32,
        input_inverse: f32,
        row_amax: &mut SharedArray<f32, ROWWISE_WARPS_PER_BLOCK>,
    ) {
        let row = thread::blockIdx_x();
        let (thread_id, lane, warp_in_block) = thread_lane_warp();
        if row >= row_count {
            return;
        }

        let (lane_in_group, group_mask, group_leader) = four_six_lane();
        let group_0 = thread_id / GROUP_THREADS as u32;
        let group_1 = group_0 + ROWWISE_GROUPS_PER_BLOCK;
        let group_2 = group_1 + ROWWISE_GROUPS_PER_BLOCK;
        let group_3 = group_2 + ROWWISE_GROUPS_PER_BLOCK;
        let group_4 = group_3 + ROWWISE_GROUPS_PER_BLOCK;
        let group_5 = group_4 + ROWWISE_GROUPS_PER_BLOCK;
        let group_6 = group_5 + ROWWISE_GROUPS_PER_BLOCK;
        let group_7 = group_6 + ROWWISE_GROUPS_PER_BLOCK;
        let values_0 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_0, lane_in_group, input_inverse);
        let values_1 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_1, lane_in_group, input_inverse);
        let values_2 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_2, lane_in_group, input_inverse);
        let values_3 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_3, lane_in_group, input_inverse);
        let values_4 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_4, lane_in_group, input_inverse);
        let values_5 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_5, lane_in_group, input_inverse);
        let values_6 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_6, lane_in_group, input_inverse);
        let values_7 =
            load_8192_cached_values::<NORMALIZE>(x, row, group_7, lane_in_group, input_inverse);
        let mut local_amax = values_0.amax();
        local_amax = max_f32(local_amax, values_1.amax());
        local_amax = max_f32(local_amax, values_2.amax());
        local_amax = max_f32(local_amax, values_3.amax());
        local_amax = max_f32(local_amax, values_4.amax());
        local_amax = max_f32(local_amax, values_5.amax());
        local_amax = max_f32(local_amax, values_6.amax());
        local_amax = max_f32(local_amax, values_7.amax());

        let tensor_amax = block_max_shared_f32_for_warps(
            row_amax,
            ROWWISE_WARPS_PER_BLOCK as u32,
            local_amax,
            lane,
            warp_in_block,
            0.0,
        );
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        if thread_id == 0 {
            unsafe {
                *amax.get_unchecked_mut(row as usize) = tensor_amax;
                *out_global_scale.get_unchecked_mut(row as usize) = global_scale;
            }
        }

        store_8192_cached_values(
            values_0,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_0,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_1,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_1,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_2,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_2,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_3,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_3,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_4,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_4,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_5,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_5,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_6,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_6,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
        store_8192_cached_values(
            values_7,
            &mut out_fp4,
            &mut out_scales,
            row,
            group_7,
            global_scale,
            scale_override,
            group_mask,
            group_leader,
            lane_in_group,
            row_count,
        );
    }

    /// The exact existing mHC RMS sum precedes the existing cached FP4 producer.
    /// Re-read original input and multiply by its inverse at each cached load,
    /// preserving the old normalized-value rounding without a global plane.
    #[inline(always)]
    pub(crate) fn mhc_rms_quant_body(
        x: &[f32],
        mut inverse_out: DisjointSlice<f32>,
        amax: DisjointSlice<f32>,
        payload: DisjointSlice<u8>,
        scales: DisjointSlice<u8>,
        global: DisjointSlice<f32>,
        row_count: u32,
        storage: &mut SharedArray<f32, ROWWISE_WARPS_PER_BLOCK>,
    ) {
        let row = thread::blockIdx_x();
        if row >= row_count {
            return;
        }
        let (tid, lane, warp) = thread_lane_warp();
        let mut square = 0.0;
        let mut j = tid;
        while j < 8192 {
            let value = x[(row * 8192 + j) as usize];
            square += value * value;
            j += 256;
        }
        let total = crate::block_reduce::block_sum_shared_f32(storage, square, lane, warp);
        let inverse = 1.0 / sqrt_f32(total / 8192.0 + 1e-6);
        if tid == 0 {
            unsafe {
                *inverse_out.get_unchecked_mut(row as usize) = inverse;
            }
        }
        rowwise_derived_amax_8192_cached_body::<true>(
            x,
            amax,
            payload,
            scales,
            global,
            row_count,
            crate::nvfp4_quant::four_six::SCALE_OVERRIDE,
            inverse,
            storage,
        );
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pack_cached_rows<const ROW_LEN: usize, const GROUPS_PER_THREAD: usize>(
        load: impl Fn(u32, usize) -> [f32; 4],
        mut amax: Option<DisjointSlice<f32>>,
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        row_count: u32,
        scale_override: f32,
        row_amax: &mut SharedArray<f32, ROWWISE_WARPS_PER_BLOCK>,
    ) {
        let row = thread::blockIdx_x();
        let (thread_id, lane, warp_in_block) = thread_lane_warp();
        if row >= row_count {
            return;
        }

        let (lane_in_group, group_mask, group_leader) = four_six_lane();
        let groups_per_row = (ROW_LEN / GROUP_SIZE) as u32;
        let group_start = thread_id / GROUP_THREADS as u32;
        let mut cached_0 = [0.0_f32; GROUPS_PER_THREAD];
        let mut cached_1 = [0.0_f32; GROUPS_PER_THREAD];
        let mut cached_2 = [0.0_f32; GROUPS_PER_THREAD];
        let mut cached_3 = [0.0_f32; GROUPS_PER_THREAD];
        let mut local_amax = 0.0_f32;
        let mut cache_index = 0usize;
        while cache_index < GROUPS_PER_THREAD {
            let group_in_row = group_start + cache_index as u32 * ROWWISE_GROUPS_PER_BLOCK;
            let column = group_in_row as usize * GROUP_SIZE + 4 * lane_in_group;
            let [value_0, value_1, value_2, value_3] = load(row, column);
            cached_0[cache_index] = value_0;
            cached_1[cache_index] = value_1;
            cached_2[cache_index] = value_2;
            cached_3[cache_index] = value_3;
            local_amax = max_f32(local_amax, abs_f32(value_0));
            local_amax = max_f32(local_amax, abs_f32(value_1));
            local_amax = max_f32(local_amax, abs_f32(value_2));
            local_amax = max_f32(local_amax, abs_f32(value_3));
            cache_index += 1;
        }

        let tensor_amax = block_max_shared_f32_for_warps(
            row_amax,
            ROWWISE_WARPS_PER_BLOCK as u32,
            local_amax,
            lane,
            warp_in_block,
            0.0,
        );
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        if thread_id == 0 {
            unsafe {
                if let Some(amax) = &mut amax {
                    *amax.get_unchecked_mut(row as usize) = tensor_amax;
                }
                *out_global_scale.get_unchecked_mut(row as usize) = global_scale;
            }
        }

        cache_index = 0;
        while cache_index < GROUPS_PER_THREAD {
            let group_in_row = group_start + cache_index as u32 * ROWWISE_GROUPS_PER_BLOCK;
            let group = row * groups_per_row + group_in_row;
            let base = group as usize * GROUP_SIZE;
            let (scale_bits, payload_word) = activation_grid_group_scale(
                cached_0[cache_index],
                cached_1[cache_index],
                cached_2[cache_index],
                cached_3[cache_index],
                global_scale,
                scale_override,
                group_mask,
                group_leader,
                lane_in_group,
            );
            unsafe {
                crate::nvfp4::csf::store_row_warp(
                    out_scales.as_mut_ptr(),
                    row_count,
                    ROW_LEN as u32,
                    row,
                    group_in_row,
                    scale_bits,
                );
                store_four_six_payload_word(
                    out_fp4.as_mut_ptr(),
                    base,
                    lane_in_group,
                    payload_word,
                );
            }
            cache_index += 1;
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

        if group_ctx.group < out_fp4.len() / 8 {
            let base = group_ctx.base as u32;
            let lane_base = 4 * group_ctx.lane as u32;
            let value_0 = padded_value(x, base + lane_base, rows, cols, padded_cols);
            let value_1 = padded_value(x, base + lane_base + 1, rows, cols, padded_cols);
            let value_2 = padded_value(x, base + lane_base + 2, rows, cols, padded_cols);
            let value_3 = padded_value(x, base + lane_base + 3, rows, cols, padded_cols);
            let out = FourSixOutputs {
                layout: 0,
                row_len: 0,
                row_count: 0,
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
                value_0,
                value_1,
                value_2,
                value_3,
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

        if group_ctx.group < out_fp4.len() / 8 {
            let out = FourSixOutputs {
                layout: 0,
                row_len: 0,
                row_count: 0,
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
        let mut linear_row_group = thread_group;
        while linear_row_group < TRANSPOSE_TILE_GROUPS {
            let row_group = tiled_row_group_without_bank_conflicts(linear_row_group);
            let tile_row = row_group / groups_per_tile_row;
            let row_col = (row_group - tile_row * groups_per_tile_row) * GROUP_SIZE;
            let lane_col = row_col + 4 * lane;
            let value_0 = unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + lane_col] };
            let value_1 = unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + lane_col + 1] };
            let value_2 = unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + lane_col + 2] };
            let value_3 = unsafe { TILE[tile_row * TRANSPOSE_TILE_STRIDE + lane_col + 3] };
            let source_row = source_row_base + tile_row;
            let group =
                source_row * groups_per_source_row + (source_col_base + row_col) / GROUP_SIZE;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane, payload_word);
            }
            linear_row_group += TRANSPOSE_GROUPS_PER_ROUND;
        }

        let groups_per_output_row = source_rows as usize / GROUP_SIZE;
        let group_in_output_row_base = source_row_base / GROUP_SIZE;
        let mut linear_transpose_group = thread_group;
        while linear_transpose_group < TRANSPOSE_TILE_GROUPS {
            let transpose_group =
                tiled_transpose_group_without_bank_conflicts(linear_transpose_group);
            let tile_row_group = transpose_group / TRANSPOSE_TILE_COLS;
            let transpose_col = transpose_group - tile_row_group * TRANSPOSE_TILE_COLS;
            let lane_row = tile_row_group * GROUP_SIZE + 4 * lane;
            let value_0 = unsafe { TILE[lane_row * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let value_1 = unsafe { TILE[(lane_row + 1) * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let value_2 = unsafe { TILE[(lane_row + 2) * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let value_3 = unsafe { TILE[(lane_row + 3) * TRANSPOSE_TILE_STRIDE + transpose_col] };
            let group_in_output_row = group_in_output_row_base + tile_row_group;
            let group =
                (source_col_base + transpose_col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if lane == 0 {
                    *transpose_out_scales.get_unchecked_mut(group) = scale_bits;
                }
                store_four_six_payload_word(
                    transpose_out_fp4.as_mut_ptr(),
                    base,
                    lane,
                    payload_word,
                );
            }
            linear_transpose_group += TRANSPOSE_GROUPS_PER_ROUND;
        }
    }

    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "one shared tile has explicit compact source geometry"
    )]
    unsafe fn stage_paired_quant_tile<const INTERIOR: bool>(
        x: *const f32,
        tile: *mut f32,
        thread_id: usize,
        row_base: usize,
        col_base: usize,
        rows: usize,
        cols: usize,
    ) {
        let load_row = thread_id / TRANSPOSE_TILE_COLS;
        let load_col = thread_id & (TRANSPOSE_TILE_COLS - 1);
        let mut load = 0;
        while load < 8 {
            let row = row_base + load_row + load * 4;
            let col = col_base + load_col;
            // Full tiles prove these bounds once per CTA. Tail tiles retain
            // logical guards; both paths use the identical source values.
            let value = if INTERIOR || (row < rows && col < cols) {
                unsafe { *x.add(row * cols + col) }
            } else {
                0.0
            };
            unsafe {
                *tile.add(packed_transpose_tile_index(load_row + load * 4, load_col)) = value;
            }
            load += 1;
        }
    }

    #[kernel]
    pub fn fp32_pair_to_nvfp4_four_six_tiled_padded_packed_scales_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales_packed: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        mut transpose_out_fp4: DisjointSlice<u8>,
        mut transpose_out_scales_packed: DisjointSlice<u8>,
        mut transpose_out_global_scale: DisjointSlice<f32>,
        mut next_gram_amax: DisjointSlice<f32>,
        source_rows: u32,
        source_cols: u32,
        padded_rows: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        static mut TILE: SharedArray<f32, PACKED_TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let source_row_base = thread::blockIdx_y() as usize * TRANSPOSE_TILE_ROWS;
        let source_col_base = thread::blockIdx_x() as usize * TRANSPOSE_TILE_COLS;
        let source_cols_usize = source_cols as usize;

        let tile_ptr = unsafe { (&mut *(&raw mut TILE)).as_mut_ptr() };
        let interior = source_row_base + TRANSPOSE_TILE_ROWS <= source_rows as usize
            && source_col_base + TRANSPOSE_TILE_COLS <= source_cols_usize;
        unsafe {
            if interior {
                stage_paired_quant_tile::<true>(
                    x.as_ptr(),
                    tile_ptr,
                    thread_id,
                    source_row_base,
                    source_col_base,
                    source_rows as usize,
                    source_cols_usize,
                );
            } else {
                stage_paired_quant_tile::<false>(
                    x.as_ptr(),
                    tile_ptr,
                    thread_id,
                    source_row_base,
                    source_col_base,
                    source_rows as usize,
                    source_cols_usize,
                );
            }
        }
        thread::sync_threads();

        let (lane, mask, leader) = four_six_lane();
        let thread_group = thread_id / GROUP_THREADS;
        let global_scale = four_six_global_scale(amax[0], scale_override);
        if source_row_base == 0 && source_col_base == 0 && thread_id == 0 {
            unsafe {
                *out_global_scale.get_unchecked_mut(0) = global_scale;
                *transpose_out_global_scale.get_unchecked_mut(0) = global_scale;
                *next_gram_amax.get_unchecked_mut(0) = 0.0;
            }
        }

        let groups_per_source_row = padded_cols as usize / GROUP_SIZE;
        let groups_per_tile_row = TRANSPOSE_TILE_COLS / GROUP_SIZE;
        let mut linear_row_group = thread_group;
        while linear_row_group < TRANSPOSE_TILE_GROUPS {
            let row_group = packed_tiled_row_group(linear_row_group);
            let tile_row = row_group / groups_per_tile_row;
            let row_col = (row_group - tile_row * groups_per_tile_row) * GROUP_SIZE;
            let lane_col = row_col + 4 * lane;
            let value_0 = unsafe { TILE[packed_transpose_tile_index(tile_row, lane_col)] };
            let value_1 = unsafe { TILE[packed_transpose_tile_index(tile_row, lane_col + 1)] };
            let value_2 = unsafe { TILE[packed_transpose_tile_index(tile_row, lane_col + 2)] };
            let value_3 = unsafe { TILE[packed_transpose_tile_index(tile_row, lane_col + 3)] };
            let source_row = source_row_base + tile_row;
            let k_group = (source_col_base + row_col) / GROUP_SIZE;
            let group = source_row * groups_per_source_row + k_group;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if lane == 0 {
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        source_row as u32,
                        k_group as u32,
                        padded_rows,
                        padded_cols,
                    );
                    *out_scales_packed.get_unchecked_mut(packed) = scale_bits;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane, payload_word);
            }
            linear_row_group += TRANSPOSE_GROUPS_PER_ROUND;
        }

        let groups_per_output_row = padded_rows as usize / GROUP_SIZE;
        let group_in_output_row_base = source_row_base / GROUP_SIZE;
        let mut linear_transpose_group = thread_group;
        while linear_transpose_group < TRANSPOSE_TILE_GROUPS {
            let transpose_group =
                tiled_transpose_group_without_bank_conflicts(linear_transpose_group);
            let tile_row_group = transpose_group / TRANSPOSE_TILE_COLS;
            let transpose_col = transpose_group - tile_row_group * TRANSPOSE_TILE_COLS;
            let lane_row = tile_row_group * GROUP_SIZE + 4 * lane;
            let value_0 = unsafe { TILE[packed_transpose_tile_index(lane_row, transpose_col)] };
            let value_1 = unsafe { TILE[packed_transpose_tile_index(lane_row + 1, transpose_col)] };
            let value_2 = unsafe { TILE[packed_transpose_tile_index(lane_row + 2, transpose_col)] };
            let value_3 = unsafe { TILE[packed_transpose_tile_index(lane_row + 3, transpose_col)] };
            let output_row = source_col_base + transpose_col;
            let group_in_output_row = group_in_output_row_base + tile_row_group;
            let group = output_row * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if lane == 0 {
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        output_row as u32,
                        group_in_output_row as u32,
                        padded_cols,
                        padded_rows,
                    );
                    *transpose_out_scales_packed.get_unchecked_mut(packed) = scale_bits;
                }
                store_four_six_payload_word(
                    transpose_out_fp4.as_mut_ptr(),
                    base,
                    lane,
                    payload_word,
                );
            }
            linear_transpose_group += TRANSPOSE_GROUPS_PER_ROUND;
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

        if group_ctx.group < out_fp4.len() / 8 {
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
                layout: 0,
                row_len: 0,
                row_count: 0,
                fp4: out_fp4,
                scales: out_scales,
                global_scale: out_global_scale,
            };
            let value_base = group_ctx.base + 4 * group_ctx.lane;
            let (value_0, value_1) = load_f32x2_global(x.as_ptr(), value_base);
            let (value_2, value_3) = load_f32x2_global(x.as_ptr(), value_base + 2);
            pack_four_six_group_values(
                bounded_amax,
                out,
                group_ctx,
                0,
                group_ctx.group == 0,
                1.0,
                value_0 * bound_scale,
                value_1 * bound_scale,
                value_2 * bound_scale,
                value_3 * bound_scale,
            );
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_padded_packed_scales_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales_packed: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        mut next_amax: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        padded_rows: u32,
        padded_cols: u32,
        scale_override: f32,
    ) {
        let ctx = four_six_group_ctx();
        let groups_per_row = padded_cols / GROUP_SIZE as u32;
        if ctx.group < padded_rows as usize * groups_per_row as usize {
            let group = ctx.group as u32;
            let row = if groups_per_row.is_power_of_two() {
                group >> groups_per_row.trailing_zeros()
            } else {
                group / groups_per_row
            };
            let k_group = group - row * groups_per_row;
            let col = k_group * GROUP_SIZE as u32 + ctx.lane as u32 * 4;
            let (v0, v1, v2, v3) = if row < rows && col < cols {
                let base = (row * cols + col) as usize;
                let (v0, v1) = load_f32x2_global(x.as_ptr(), base);
                let (v2, v3) = load_f32x2_global(x.as_ptr(), base + 2);
                (v0, v1, v2, v3)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            let global = four_six_global_scale(amax[0], scale_override);
            let (scale, payload) = six_grid_group_scale(
                v0,
                v1,
                v2,
                v3,
                global,
                scale_override,
                ctx.mask,
                ctx.leader,
                ctx.lane,
            );
            unsafe {
                if ctx.group == 0 && ctx.lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global;
                    *next_amax.get_unchecked_mut(0) = 0.0;
                }
                if ctx.lane == 0 {
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        row as u32,
                        k_group as u32,
                        padded_rows,
                        padded_cols,
                    );
                    *out_scales_packed.get_unchecked_mut(packed) = scale;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), ctx.base, ctx.lane, payload);
            }
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_exact_bounded_amax_packed_scales_kernel(
        x: &[f32],
        original_amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales_packed: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        apply_bound: u32,
    ) {
        let group_ctx = four_six_group_ctx();
        let group_count = rows as usize * cols as usize / GROUP_SIZE;

        if group_ctx.group < group_count {
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
            let value_base = group_ctx.base + 4 * group_ctx.lane;
            let (value_0, value_1) = load_f32x2_global(x.as_ptr(), value_base);
            let (value_2, value_3) = load_f32x2_global(x.as_ptr(), value_base + 2);
            let global_scale = four_six_global_scale(bounded_amax, 1.0);
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0 * bound_scale,
                value_1 * bound_scale,
                value_2 * bound_scale,
                value_3 * bound_scale,
                global_scale,
                1.0,
                group_ctx.mask,
                group_ctx.leader,
                group_ctx.lane,
            );
            unsafe {
                if group_ctx.group == 0 && group_ctx.lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if group_ctx.lane == 0 {
                    let k_groups = cols as usize / GROUP_SIZE;
                    let row = group_ctx.group / k_groups;
                    let k_group = group_ctx.group - row * k_groups;
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        row as u32,
                        k_group as u32,
                        rows,
                        cols,
                    );
                    *out_scales_packed.get_unchecked_mut(packed) = scale_bits;
                }
                store_four_six_payload_word(
                    out_fp4.as_mut_ptr(),
                    group_ctx.base,
                    group_ctx.lane,
                    payload_word,
                );
            }
        }
    }

    #[kernel]
    pub fn fp32_to_nvfp4_four_six_exact_lazy_bounded_amax_packed_scales_rebase_and_clear_source_amax_kernel(
        x: &[f32],
        bound_amax: &[f32],
        mut rebase_original_amax: DisjointSlice<f32>,
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales_packed: DisjointSlice<u8>,
        mut out_global_scale: DisjointSlice<f32>,
        mut rebase_out_global_scale: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
    ) {
        let group_ctx = four_six_group_ctx();
        let group_count = rows as usize * cols as usize / GROUP_SIZE;

        if group_ctx.group < group_count {
            let bound = bound_amax[0];
            let bounded_amax = if bound > 1.0 {
                bound * (1.0 / bound)
            } else {
                bound
            };
            let bound_scale = if bound > 1.0 { 1.0 / bound } else { 1.0 };
            let value_base = group_ctx.base + 4 * group_ctx.lane;
            let (value_0, value_1) = load_f32x2_global(x.as_ptr(), value_base);
            let (value_2, value_3) = load_f32x2_global(x.as_ptr(), value_base + 2);
            let global_scale = four_six_global_scale(bounded_amax, 1.0);
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0 * bound_scale,
                value_1 * bound_scale,
                value_2 * bound_scale,
                value_3 * bound_scale,
                global_scale,
                1.0,
                group_ctx.mask,
                group_ctx.leader,
                group_ctx.lane,
            );
            unsafe {
                if group_ctx.group == 0 && group_ctx.lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                    let sqrt_bound_scale = if bound > 1.0 {
                        1.0 / sqrt_f32(bound)
                    } else {
                        1.0
                    };
                    // This is the only reader of source amax in this kernel.
                    // Consume it before initializing the following factor max.
                    let source_amax = *rebase_original_amax.as_mut_ptr();
                    *rebase_out_global_scale.get_unchecked_mut(0) =
                        four_six_global_scale(source_amax * sqrt_bound_scale, 1.0);
                    *rebase_original_amax.get_unchecked_mut(0) = 0.0;
                }
                if group_ctx.lane == 0 {
                    let k_groups = cols as usize / GROUP_SIZE;
                    let row = group_ctx.group / k_groups;
                    let k_group = group_ctx.group - row * k_groups;
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        row as u32,
                        k_group as u32,
                        rows,
                        cols,
                    );
                    *out_scales_packed.get_unchecked_mut(packed) = scale_bits;
                }
                store_four_six_payload_word(
                    out_fp4.as_mut_ptr(),
                    group_ctx.base,
                    group_ctx.lane,
                    payload_word,
                );
            }
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

        if group_ctx.group < out_fp4.len() / 8 {
            let base = group_ctx.base as u32;
            let lane_base = 4 * group_ctx.lane as u32;
            let value_0 =
                transposed_padded_value(x, base + lane_base, source_rows, source_cols, padded_cols);
            let value_1 = transposed_padded_value(
                x,
                base + lane_base + 1,
                source_rows,
                source_cols,
                padded_cols,
            );
            let value_2 = transposed_padded_value(
                x,
                base + lane_base + 2,
                source_rows,
                source_cols,
                padded_cols,
            );
            let value_3 = transposed_padded_value(
                x,
                base + lane_base + 3,
                source_rows,
                source_cols,
                padded_cols,
            );
            let out = FourSixOutputs {
                layout: 0,
                row_len: 0,
                row_count: 0,
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
                value_0,
                value_1,
                value_2,
                value_3,
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

        if group_ctx.group < out_fp4.len() / 8 {
            let base = group_ctx.base as u32;
            let lane_base = 4 * group_ctx.lane as u32;
            let value_0 = transposed_exact_value(x, base + lane_base, source_rows, source_cols);
            let value_1 = transposed_exact_value(x, base + lane_base + 1, source_rows, source_cols);
            let value_2 = transposed_exact_value(x, base + lane_base + 2, source_rows, source_cols);
            let value_3 = transposed_exact_value(x, base + lane_base + 3, source_rows, source_cols);
            let out = FourSixOutputs {
                layout: 0,
                row_len: 0,
                row_count: 0,
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
                value_0,
                value_1,
                value_2,
                value_3,
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

        if group_ctx.group < out_fp4.len() / 8 {
            let base = group_ctx.base as u32;
            let lane_base = 4 * group_ctx.lane as u32;
            let value_0 = transposed_exact_value_pow2(
                x,
                base + lane_base,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let value_1 = transposed_exact_value_pow2(
                x,
                base + lane_base + 1,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let value_2 = transposed_exact_value_pow2(
                x,
                base + lane_base + 2,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let value_3 = transposed_exact_value_pow2(
                x,
                base + lane_base + 3,
                source_rows_shift,
                source_rows_mask,
                source_cols,
            );
            let out = FourSixOutputs {
                layout: 0,
                row_len: 0,
                row_count: 0,
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
                value_0,
                value_1,
                value_2,
                value_3,
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
        let group_in_output_row_base = source_row_base / GROUP_SIZE;
        let global_scale = four_six_global_scale(amax[0], scale_override);
        let mut linear_group = thread_group;

        while linear_group < TRANSPOSE_TILE_GROUPS {
            let transpose_group = tiled_transpose_group_without_bank_conflicts(linear_group);
            let tile_row_group = transpose_group / TRANSPOSE_TILE_COLS;
            let col = transpose_group - tile_row_group * TRANSPOSE_TILE_COLS;
            let lane_row = tile_row_group * GROUP_SIZE + 4 * lane;
            let value_0 = unsafe { TILE[lane_row * TRANSPOSE_TILE_STRIDE + col] };
            let value_1 = unsafe { TILE[(lane_row + 1) * TRANSPOSE_TILE_STRIDE + col] };
            let value_2 = unsafe { TILE[(lane_row + 2) * TRANSPOSE_TILE_STRIDE + col] };
            let value_3 = unsafe { TILE[(lane_row + 3) * TRANSPOSE_TILE_STRIDE + col] };
            let group_in_output_row = group_in_output_row_base + tile_row_group;
            let group = (source_col_base + col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if group == 0 && lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane, payload_word);
            }

            linear_group += TRANSPOSE_GROUPS_PER_ROUND;
        }
    }

    #[kernel]
    pub fn fp32_transpose_to_nvfp4_four_six_exact_pow2_tiled_packed_scales_kernel(
        x: &[f32],
        amax: &[f32],
        mut out_fp4: DisjointSlice<u8>,
        mut out_scales_packed: DisjointSlice<u8>,
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
        let group_in_output_row_base = source_row_base / GROUP_SIZE;
        let global_scale = four_six_global_scale(amax[0], scale_override);
        let mut linear_group = thread_group;

        while linear_group < TRANSPOSE_TILE_GROUPS {
            let transpose_group = tiled_transpose_group_without_bank_conflicts(linear_group);
            let tile_row_group = transpose_group / TRANSPOSE_TILE_COLS;
            let col = transpose_group - tile_row_group * TRANSPOSE_TILE_COLS;
            let lane_row = tile_row_group * GROUP_SIZE + 4 * lane;
            let value_0 = unsafe { TILE[lane_row * TRANSPOSE_TILE_STRIDE + col] };
            let value_1 = unsafe { TILE[(lane_row + 1) * TRANSPOSE_TILE_STRIDE + col] };
            let value_2 = unsafe { TILE[(lane_row + 2) * TRANSPOSE_TILE_STRIDE + col] };
            let value_3 = unsafe { TILE[(lane_row + 3) * TRANSPOSE_TILE_STRIDE + col] };
            let output_row = source_col_base + col;
            let group_in_output_row = group_in_output_row_base + tile_row_group;
            let group = output_row * (source_rows as usize / GROUP_SIZE) + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                mask,
                leader,
                lane,
            );
            unsafe {
                if group == 0 && lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if lane == 0 {
                    let packed = Sm120ScaleLayout::block_major_byte_offset(
                        output_row as u32,
                        group_in_output_row as u32,
                        source_cols,
                        source_rows,
                    );
                    *out_scales_packed.get_unchecked_mut(packed) = scale_bits;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane, payload_word);
            }

            linear_group += TRANSPOSE_GROUPS_PER_ROUND;
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
        static mut TILE: SharedArray<f32, BOUNDED_TRANSPOSE_TILE_ELEMS> = SharedArray::UNINIT;

        let thread_id = thread::threadIdx_x() as usize;
        let source_row_base = thread::blockIdx_y() as usize * BOUNDED_TRANSPOSE_TILE_ROWS;
        let source_col_base = thread::blockIdx_x() as usize * TRANSPOSE_TILE_COLS;
        let source_cols_usize = source_cols as usize;

        let mut load_offset = thread_id;
        while load_offset < BOUNDED_TRANSPOSE_TILE_LOAD_ELEMS {
            let row = load_offset / TRANSPOSE_TILE_COLS;
            let col = load_offset - row * TRANSPOSE_TILE_COLS;
            unsafe {
                TILE[row * BOUNDED_TRANSPOSE_TILE_STRIDE + col] =
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
            let lane_row = 4 * lane;
            let mut value_0 = unsafe { TILE[lane_row * BOUNDED_TRANSPOSE_TILE_STRIDE + col] };
            let mut value_1 = unsafe { TILE[(lane_row + 1) * BOUNDED_TRANSPOSE_TILE_STRIDE + col] };
            let mut value_2 = unsafe { TILE[(lane_row + 2) * BOUNDED_TRANSPOSE_TILE_STRIDE + col] };
            let mut value_3 = unsafe { TILE[(lane_row + 3) * BOUNDED_TRANSPOSE_TILE_STRIDE + col] };
            if apply_bound != 0 {
                value_0 *= scale;
                value_1 *= scale;
                value_2 *= scale;
                value_3 *= scale;
            }
            let group = (source_col_base + col) * groups_per_output_row + group_in_output_row;
            let base = group * GROUP_SIZE;
            let (scale_bits, payload_word) = six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                1.0,
                mask,
                leader,
                lane,
            );
            unsafe {
                if group == 0 && lane == 0 {
                    *out_global_scale.get_unchecked_mut(0) = global_scale;
                }
                if lane == 0 {
                    *out_scales.get_unchecked_mut(group) = scale_bits;
                }
                store_four_six_payload_word(out_fp4.as_mut_ptr(), base, lane, payload_word);
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
        let value_base = group_ctx.base + 4 * group_ctx.lane;
        let (value_0, value_1) = load_f32x2_global(x.as_ptr(), value_base);
        let (value_2, value_3) = load_f32x2_global(x.as_ptr(), value_base + 2);
        pack_four_six_group_values(
            amax[row],
            out,
            group_ctx,
            row,
            writes_global_scale,
            scale_override,
            value_0,
            value_1,
            value_2,
            value_3,
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
        value_0: f32,
        value_1: f32,
        value_2: f32,
        value_3: f32,
    ) {
        let global_scale = four_six_global_scale(tensor_amax, scale_override);
        let (scale_bits, payload_word) = if out.layout == 1 {
            activation_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                group_ctx.mask,
                group_ctx.leader,
                group_ctx.lane,
            )
        } else {
            six_grid_group_scale(
                value_0,
                value_1,
                value_2,
                value_3,
                global_scale,
                scale_override,
                group_ctx.mask,
                group_ctx.leader,
                group_ctx.lane,
            )
        };
        unsafe {
            if writes_global_scale && group_ctx.lane == 0 {
                *out.global_scale.get_unchecked_mut(scale_row) = global_scale;
            }
            if out.layout == 2 {
                crate::nvfp4::csf::store_warp(out.scales.as_mut_ptr(), group_ctx.group, scale_bits);
            } else if out.layout == 1 {
                crate::nvfp4::csf::store_row_warp(
                    out.scales.as_mut_ptr(),
                    out.row_count,
                    out.row_len.div_ceil(128) * 128,
                    scale_row as u32,
                    (group_ctx.group as u32) % (out.row_len / 16),
                    scale_bits,
                );
            } else if group_ctx.lane == 0 {
                *out.scales.get_unchecked_mut(group_ctx.group) = scale_bits;
            }
            store_four_six_payload_word(
                out.fp4.as_mut_ptr(),
                group_ctx.base,
                group_ctx.lane,
                payload_word,
            );
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
