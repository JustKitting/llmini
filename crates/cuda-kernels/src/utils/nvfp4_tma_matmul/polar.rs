use super::{
    Nvfp4GemmParams, ROUTE_FEATURE_M_TOKEN_K, ROUTE_FEATURE_N_TOKEN_K, ROUTE_NONE,
    ROUTE_TOKEN_M_FEATURE_K, ROUTE_TOKEN_M_FEATURE_N,
};

include!(concat!(env!("OUT_DIR"), "/nvfp4_polar_config.rs"));
include!("geometry.rs");

#[cuda_module]
pub mod module {
    use super::*;
    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_polar_step_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        source: &[f32],
        mut out: DisjointSlice<f32>,
        bound_amax: &[f32],
        output_amax: &[f32],
        params: Nvfp4GemmParams,
        a_scale: f32,
        logical_rows: u32,
        logical_cols: u32,
        apply_bound: u32,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut OUTPUT_AMAX_REDUCE: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> =
            SharedArray::UNINIT;

        let tile = CtaTile::new(thread_id);
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };

        if thread_id == 0 {
            prefetch_tma_descriptor(a_tma);
            prefetch_tma_descriptor(b_tma);
            prefetch_tma_descriptor(a_scale_tma);
            prefetch_tma_descriptor(b_scale_tma);
            unsafe {
                let mut stage = 0;
                while stage < TMA_PIPELINE_STAGES {
                    mbarrier_init(stage_barrier(tma_bars, stage), 1);
                    mbarrier_init(stage_barrier(empty_bars, stage), MMA_THREADS_PER_BLOCK);
                    stage += 1;
                }
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        let bound = bound_amax[0];
        let bound_scale = if apply_bound != 0 && bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let inputs = PolarStepInputs {
            source,
            rows: logical_rows,
            cols: logical_cols,
            bound_scale,
            a_scale,
        };
        let mut local_amax = 0.0;
        run_tma_nvfp4_full_tile_polar_step_amax!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            tile,
            params,
            a_packs_base,
            b_packs_base,
            a_scales_base,
            b_scales_base,
            tma_bars,
            empty_bars,
            inputs,
            &mut out,
            &mut local_amax,
        );

        let (_, lane, warp_in_block) = thread_lane_warp();
        if let Some(cta_amax) = unsafe {
            block_max_leader_f32(&mut OUTPUT_AMAX_REDUCE, local_amax, lane, warp_in_block)
        } {
            atomic_max_nonnegative_f32(output_amax, cta_amax);
        }
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_polar_step_row_sumsq_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        source: &[f32],
        mut out: DisjointSlice<f32>,
        bound_amax: &[f32],
        mut row_sumsq_partials: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
        a_scale: f32,
        logical_rows: u32,
        logical_cols: u32,
        apply_bound: u32,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut ROW_SUMS: SharedArray<f32, POLAR_ROW_SUMS_PER_CTA> = SharedArray::UNINIT;

        let tile = CtaTile::new(thread_id);
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };
        let row_sums = unsafe { (&mut *(&raw mut ROW_SUMS)).as_mut_ptr() };

        if thread_id == 0 {
            prefetch_tma_descriptor(a_tma);
            prefetch_tma_descriptor(b_tma);
            prefetch_tma_descriptor(a_scale_tma);
            prefetch_tma_descriptor(b_scale_tma);
            unsafe {
                let mut stage = 0;
                while stage < TMA_PIPELINE_STAGES {
                    mbarrier_init(stage_barrier(tma_bars, stage), 1);
                    mbarrier_init(stage_barrier(empty_bars, stage), MMA_THREADS_PER_BLOCK);
                    stage += 1;
                }
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        let bound = bound_amax[0];
        let bound_scale = if apply_bound != 0 && bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let inputs = PolarStepInputs {
            source,
            rows: logical_rows,
            cols: logical_cols,
            bound_scale,
            a_scale,
        };
        let mut local_row_sums = [0.0_f32; POLAR_ROW_SUMS_PER_THREAD];
        run_tma_nvfp4_full_tile_polar_step_row_sumsq!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            tile,
            params,
            a_packs_base,
            b_packs_base,
            a_scales_base,
            b_scales_base,
            tma_bars,
            empty_bars,
            inputs,
            &mut out,
            &mut local_row_sums,
        );

        if thread_id < MMA_THREADS_PER_BLOCK {
            let lane = thread_id & 31;
            let group_mask = 0x0f_u32 << (lane & !3);
            let mut m_repeat = 0;
            while m_repeat < M_REPEAT {
                let sum0 = eighth_warp_sum_f32(local_row_sums[(m_repeat * 2) as usize], group_mask);
                let sum1 =
                    eighth_warp_sum_f32(local_row_sums[(m_repeat * 2 + 1) as usize], group_mask);
                if tile.thread_in_group == 0 {
                    let local_row0 = tile.warp_m * M_PER_WARP + m_repeat * MMA_M + tile.group;
                    let local_row1 = local_row0 + 8;
                    unsafe {
                        *row_sums.add((local_row0 * WARP_TILES_N + tile.warp_n) as usize) = sum0;
                        *row_sums.add((local_row1 * WARP_TILES_N + tile.warp_n) as usize) = sum1;
                    }
                }
                m_repeat += 1;
            }
        }
        thread::sync_threads();

        if thread_id < TILE_M && thread::blockIdx_y() * TILE_M + thread_id < logical_rows {
            let mut sum = 0.0_f32;
            let mut warp_n = 0;
            while warp_n < WARP_TILES_N {
                sum += unsafe { *row_sums.add((thread_id * WARP_TILES_N + warp_n) as usize) };
                warp_n += 1;
            }
            let row = thread::blockIdx_y() * TILE_M + thread_id;
            let partial = row * thread::gridDim_x() + thread::blockIdx_x();
            unsafe {
                *row_sumsq_partials.get_unchecked_mut(partial as usize) = sum;
            }
        }
    }
}
