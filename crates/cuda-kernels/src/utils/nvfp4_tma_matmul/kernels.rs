#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nvfp4GemmParams {
    pub token_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub global_scale_mode: u32,
    pub weight_global_scale: f32,
    pub a_global_scale: u64,
    pub b_global_scale: u64,
    pub route_mode: u32,
    pub route_token_tiles: u32,
    pub route_feature_tiles: u32,
    pub route_active_tiles: u32,
    pub route_masks: u64,
}

pub const ROUTE_NONE: u32 = 0;
pub const ROUTE_TOKEN_M_FEATURE_N: u32 = 1;
pub const ROUTE_TOKEN_M_FEATURE_K: u32 = 2;
pub const ROUTE_FEATURE_M_TOKEN_K: u32 = 3;
pub const ROUTE_FEATURE_N_TOKEN_K: u32 = 4;

#[path = "polar.rs"]
pub mod polar;

include!(concat!(env!("OUT_DIR"), "/nvfp4_config.rs"));
include!("geometry.rs");

macro_rules! run_affine_tile {
    (
        $a_tma:ident,
        $b_tma:ident,
        $a_scale_tma:ident,
        $b_scale_tma:ident,
        $out:ident,
        $bias_bytes:ident,
        $bias_scales:ident,
        $bias_global_scale:ident,
        $params:ident,
        $output_tile:expr,
        $output_stride:expr,
        $output_offset:expr $(,)?
    ) => {{
        $params.global_scale_mode = 2;
        $params.route_mode = ROUTE_NONE;
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: CsfAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: CsfBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut BIAS_SM: BiasSmem = SharedArray::UNINIT;

        let tile = CtaTile::new_at(
            thread_id,
            N_LAYOUT_WARP_CONTIGUOUS,
            thread::blockIdx_y(),
            $output_tile,
        );
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };
        let bias_tile = unsafe { &mut *(&raw mut BIAS_SM) };

        if thread_id < TILE_N / 2 {
            let bias_col = tile.col_base + thread_id * 2;
            let (bias0, bias1) = if bias_col + 1 < $params.output_dim {
                nvfp4_values2(
                    $bias_bytes,
                    $bias_scales,
                    $bias_global_scale[0],
                    bias_col as usize,
                )
            } else {
                (0.0, 0.0)
            };
            store_f32x2_shared(
                bias_tile.as_mut_ptr(),
                (thread_id * 2) as usize,
                bias0,
                bias1,
            );
        }

        if thread_id == 0 {
            prefetch_tma_descriptor($a_tma);
            prefetch_tma_descriptor($b_tma);
            prefetch_tma_descriptor($a_scale_tma);
            prefetch_tma_descriptor($b_scale_tma);
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

        run_tma_nvfp4_full_tile_affine!(
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            tile,
            $params,
            a_packs_base,
            b_packs_base,
            a_scales_base,
            b_scales_base,
            tma_bars,
            empty_bars,
            &mut $out,
            bias_tile,
            $output_stride,
            $output_offset,
        );
    }};
}

// Expand in each entry so the device compiler keeps the tile pipeline in the kernel.
macro_rules! run_residual_tile {
    (
        $a_tma:ident,
        $b_tma:ident,
        $a_scale_tma:ident,
        $b_scale_tma:ident,
        $residual:ident,
        $bias_bytes:ident,
        $bias_scales:ident,
        $bias_global_scale:ident,
        $params:ident,
        $add_residual:literal $(,)?
    ) => {{
        $params.global_scale_mode = 2;
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: ResidualBarrierStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: ResidualBarrierStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: ResidualScalesStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: ResidualScalesStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: ResidualAPacksStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: ResidualBPacksStages = SharedArray::UNINIT;
        static mut BIAS_SM: BiasSmem = SharedArray::UNINIT;

        let tile = CtaTile::new_csf(thread_id);
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };
        let bias_tile = unsafe { &mut *(&raw mut BIAS_SM) };

        if thread_id < TILE_N / 2 {
            let bias_col = thread::blockIdx_x() * TILE_N + thread_id * 2;
            let (bias0, bias1) = nvfp4_values2(
                $bias_bytes,
                $bias_scales,
                $bias_global_scale[0],
                bias_col as usize,
            );
            store_f32x2_shared(
                bias_tile.as_mut_ptr(),
                (thread_id * 2) as usize,
                bias0,
                bias1,
            );
        }

        if thread_id == 0 {
            prefetch_tma_descriptor($a_tma);
            prefetch_tma_descriptor($b_tma);
            prefetch_tma_descriptor($a_scale_tma);
            prefetch_tma_descriptor($b_scale_tma);
            unsafe {
                let mut stage = 0;
                while stage < 2 {
                    mbarrier_init(stage_barrier(tma_bars, stage), 1);
                    mbarrier_init(stage_barrier(empty_bars, stage), MMA_THREADS_PER_BLOCK);
                    stage += 1;
                }
                fence_proxy_async_shared_cta();
            }
        }
        thread::sync_threads();

        run_tma_nvfp4_full_tile_residual!(
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            tile,
            $params,
            a_packs_base,
            b_packs_base,
            a_scales_base,
            b_scales_base,
            tma_bars,
            empty_bars,
            &mut $residual,
            bias_tile,
            $add_residual,
        );
    }};
}

#[cuda_module]
pub mod affine_columns {
    use super::*;

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_affine_columns_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
        first_output_tile: u32,
    ) {
        run_affine_tile!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            bias_bytes,
            bias_scales,
            bias_global_scale,
            params,
            first_output_tile + thread::blockIdx_x(),
            params.output_dim,
            0,
        );
    }
}

#[cuda_module]
pub mod affine_strided {
    use super::*;

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_affine_strided_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
        output_stride: u32,
        output_offset: u32,
    ) {
        run_affine_tile!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            bias_bytes,
            bias_scales,
            bias_global_scale,
            params,
            thread::blockIdx_x(),
            output_stride,
            output_offset,
        );
    }
}

#[cuda_module]
pub mod module {
    use super::*;

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 2))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4GemmParams,
    ) {
        params.global_scale_mode = 2;
        params.route_mode = ROUTE_NONE;
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new(thread_id);
        if !routed_output_tile_active(tile, params) {
            zero_output_tile_f32(tile, params, &mut out);
            return;
        }
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };

        if thread_id == 0 {
            // Matches CUTLASS SM120: one elected thread prefetches mainloop tensor maps before setup.
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

        run_tma_nvfp4_full_tile!(
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
            &mut out,
        );
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 2))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_exact_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new(thread_id);
        if !routed_output_tile_active(tile, params) {
            zero_output_tile_f32(tile, params, &mut out);
            return;
        }
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

        run_tma_nvfp4_full_tile_exact!(
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
            &mut out,
        );
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 2))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        output_amax: &[f32],
        params: Nvfp4GemmParams,
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

        let mut local_amax = 0.0;
        run_tma_nvfp4_full_tile_amax!(
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
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 2))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_symmetric_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        output_amax: &[f32],
        params: Nvfp4GemmParams,
    ) {
        let subtile = thread::blockIdx_x() % (TILE_M / TILE_N);
        let cta = thread::blockIdx_x() / (TILE_M / TILE_N);
        let tiles = params.output_dim / TILE_M;
        let mut tile_row = 0;
        let mut tile_col_offset = cta;
        let mut row_width = tiles;
        while tile_col_offset >= row_width {
            tile_col_offset -= row_width;
            tile_row += 1;
            row_width -= 1;
        }
        let tile_col = tile_row + tile_col_offset;

        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut OUTPUT_AMAX_REDUCE: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> =
            SharedArray::UNINIT;

        let tile = CtaTile::new_at(
            thread_id,
            NVFP4_N_LAYOUT,
            tile_row,
            tile_col * (TILE_M / TILE_N) + subtile,
        );
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

        let mirror = tile_row != tile_col;
        let mut local_amax = 0.0;
        run_tma_nvfp4_full_tile_symmetric_amax!(
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
            &mut out,
            &mut local_amax,
            mirror,
        );

        let (_, lane, warp_in_block) = thread_lane_warp();
        if let Some(cta_amax) = unsafe {
            block_max_leader_f32(&mut OUTPUT_AMAX_REDUCE, local_amax, lane, warp_in_block)
        } {
            atomic_max_nonnegative_f32(output_amax, cta_amax);
        }
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 2))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_gram_factor_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        gram: &[f32],
        bound_amax: &[f32],
        logical_dim: u32,
        b_scale: f32,
        c_scale: f32,
        apply_bound: u32,
        mut out: DisjointSlice<f32>,
        output_amax: &[f32],
        params: Nvfp4GemmParams,
    ) {
        let subtile = thread::blockIdx_x() % (TILE_M / TILE_N);
        let cta = thread::blockIdx_x() / (TILE_M / TILE_N);
        let tiles = params.output_dim / TILE_M;
        let mut tile_row = 0;
        let mut tile_col_offset = cta;
        let mut row_width = tiles;
        while tile_col_offset >= row_width {
            tile_col_offset -= row_width;
            tile_row += 1;
            row_width -= 1;
        }
        let tile_col = tile_row + tile_col_offset;

        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: NativeAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: NativeBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut OUTPUT_AMAX_REDUCE: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> =
            SharedArray::UNINIT;

        let tile = CtaTile::new_at(
            thread_id,
            NVFP4_N_LAYOUT,
            tile_row,
            tile_col * (TILE_M / TILE_N) + subtile,
        );
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
        let inputs = GramFactorInputs {
            gram,
            dim: logical_dim,
            b_scale,
            c_scale,
            bound_scale: if apply_bound != 0 && bound > 1.0 {
                1.0 / bound
            } else {
                1.0
            },
        };
        let mirror = tile_row != tile_col;
        let mut local_amax = 0.0;
        run_tma_nvfp4_full_tile_gram_factor_amax!(
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
            mirror,
        );

        let (_, lane, warp_in_block) = thread_lane_warp();
        if let Some(cta_amax) = unsafe {
            block_max_leader_f32(&mut OUTPUT_AMAX_REDUCE, local_amax, lane, warp_in_block)
        } {
            atomic_max_nonnegative_f32(output_amax, cta_amax);
        }
    }

    #[kernel]
    pub fn nvfp4_gemm_tma_row_sumsq_reduce_kernel(
        row_sumsq_partials: &[f32],
        output_amax: &[f32],
        rows: u32,
        partials_per_row: u32,
    ) {
        static mut WARP_SUMS: SharedArray<f32, 2> = SharedArray::UNINIT;

        let row = thread::blockIdx_x();
        if row >= rows {
            return;
        }
        let (thread_id, lane, warp) = thread_lane_warp();
        let base = row * partials_per_row;
        let mut local = 0.0_f32;
        let mut partial = thread_id;
        while partial < partials_per_row {
            local += row_sumsq_partials[(base + partial) as usize];
            partial += thread::blockDim_x();
        }
        let sum = unsafe { block_sum_shared_f32(&mut WARP_SUMS, local, lane, warp) };
        if thread_id == 0 {
            let output = unsafe { &*(output_amax.as_ptr() as *const DeviceAtomicU32) };
            output.fetch_max(sum.to_bits(), AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_affine_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
    ) {
        run_affine_tile!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            bias_bytes,
            bias_scales,
            bias_global_scale,
            params,
            thread::blockIdx_x(),
            params.output_dim,
            0,
        );
    }

    #[kernel]
    #[launch_bounds(256, 2)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_residual_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut residual: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
    ) {
        run_residual_tile!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            residual,
            bias_bytes,
            bias_scales,
            bias_global_scale,
            params,
            true,
        );
    }

    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn nvfp4_gemm_tma_branch_output_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
    ) {
        run_residual_tile!(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            bias_bytes,
            bias_scales,
            bias_global_scale,
            params,
            false,
        );
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_relu2_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut pre_activation: DisjointSlice<f32>,
        pre_activation_f16: *mut u16,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
    ) {
        params.global_scale_mode = 2;
        params.route_mode = ROUTE_NONE;
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: CsfAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: CsfBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new_csf(thread_id);
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

        if pre_activation_f16.is_null() {
            run_tma_nvfp4_full_tile_relu2!(
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
                &mut pre_activation,
                &mut out,
                bias_bytes,
                bias_scales,
                bias_global_scale[0],
            );
        } else {
            run_tma_nvfp4_full_tile_relu2_f16!(
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
                &mut pre_activation,
                pre_activation_f16,
                &mut out,
                bias_bytes,
                bias_scales,
                bias_global_scale[0],
            );
        }
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_relu2_compact_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        pre_activation_f16: *mut u16,
        mut out: DisjointSlice<f32>,
        bias_bytes: &[u8],
        bias_scales: &[u8],
        bias_global_scale: &[f32],
        mut params: Nvfp4GemmParams,
    ) {
        params.global_scale_mode = 2;
        params.route_mode = ROUTE_NONE;
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: CsfAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: CsfBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut BIAS_SM: BiasSmem = SharedArray::UNINIT;

        let tile = CtaTile::new_csf(thread_id);
        let tma_bars = unsafe { (&mut *(&raw mut TMA_BARS)).as_mut_ptr() };
        let empty_bars = unsafe { (&mut *(&raw mut EMPTY_BARS)).as_mut_ptr() };
        let a_scales_base = unsafe { (&mut *(&raw mut A_SCALES_SM)).as_mut_ptr() };
        let b_scales_base = unsafe { (&mut *(&raw mut B_SCALES_SM)).as_mut_ptr() };
        let a_packs_base = unsafe { (&mut *(&raw mut A_PACKS_SM)).as_mut_ptr() };
        let b_packs_base = unsafe { (&mut *(&raw mut B_PACKS_SM)).as_mut_ptr() };
        let bias_tile = unsafe { &mut *(&raw mut BIAS_SM) };

        if thread_id < TILE_N / 2 {
            let bias_col = thread::blockIdx_x() * TILE_N + thread_id * 2;
            let (bias0, bias1) = nvfp4_values2(
                bias_bytes,
                bias_scales,
                bias_global_scale[0],
                bias_col as usize,
            );
            store_f32x2_shared(
                bias_tile.as_mut_ptr(),
                (thread_id * 2) as usize,
                bias0,
                bias1,
            );
        }

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

        run_tma_nvfp4_full_tile_relu2_compact!(
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
            pre_activation_f16,
            &mut out,
            bias_tile,
        );
    }
    #[inline(always)]
    fn csf_exact_body<const ROUTED: bool, const FUSED: bool>(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        mut params: Nvfp4GemmParams,
        gradient: GemmGradientTransform,
    ) {
        if !ROUTED {
            params.route_mode = ROUTE_NONE;
        }
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: CsfAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: CsfBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new_csf(thread_id);
        if !FUSED && !routed_output_tile_active(tile, params) {
            zero_output_tile_f32(tile, params, &mut out);
            return;
        }
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

        if FUSED {
            static mut EXP_SUMS: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> = SharedArray::UNINIT;
            static mut LIN_SUMS: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> = SharedArray::UNINIT;
            static mut CURV_SUMS: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> = SharedArray::UNINIT;
            static mut NORM_SUMS: SharedArray<f32, TMA_NVFP4_WARPS_PER_BLOCK> = SharedArray::UNINIT;
            let mut sums = GradientSums::default();
            run_csf_gradient!(
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
                &mut out,
                gradient,
                &mut sums
            );
            if gradient.learn_scales != 0 {
                if let Some((exponential, linear, curvature, norm)) = unsafe {
                    block_sum_quad_leader_f32_for_warps(
                        &mut EXP_SUMS,
                        &mut LIN_SUMS,
                        &mut CURV_SUMS,
                        &mut NORM_SUMS,
                        TMA_NVFP4_WARPS_PER_BLOCK as u32,
                        sums.exponential,
                        sums.linear,
                        sums.curvature,
                        sums.norm,
                        thread_id % 32,
                        thread_id / 32,
                    )
                } {
                    let cta = thread::blockIdx_y() * thread::gridDim_x() + thread::blockIdx_x();
                    unsafe {
                        store_sums(gradient, cta, exponential, linear, curvature, norm);
                    }
                }
            }
        } else {
            run_csf_exact!(
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
                &mut out,
            );
        }
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    pub fn nvfp4_gemm_tma_csf_exact_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        out: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        csf_exact_body::<true, false>(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            params,
            GemmGradientTransform::default(),
        );
    }
    #[kernel]
    #[launch_bounds(288, 1)]
    pub fn nvfp4_gemm_tma_csf_dense_exact_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        out: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        csf_exact_body::<false, false>(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            params,
            GemmGradientTransform::default(),
        );
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    pub fn nvfp4_gemm_tma_csf_gradient_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        out: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
        gradient: GemmGradientTransform,
    ) {
        csf_exact_body::<true, true>(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            params,
            gradient,
        );
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    pub fn nvfp4_gemm_tma_csf_dense_gradient_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        out: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
        gradient: GemmGradientTransform,
    ) {
        csf_exact_body::<false, true>(
            a_tma,
            b_tma,
            a_scale_tma,
            b_scale_tma,
            out,
            params,
            gradient,
        );
    }

    #[kernel]
    #[launch_bounds(288, 1)]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_csf_relu2_backward_f16_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        pre_activation: &[u16],
        mut out: DisjointSlice<f32>,
        mut output_chunk_amax: DisjointSlice<f32>,
        mut params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: CsfAScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: CsfBScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new_csf(thread_id);
        if !routed_output_tile_active(tile, params) {
            zero_output_tile_f32(tile, params, &mut out);
            if thread_id < WARPS_PER_BLOCK {
                let cta =
                    thread::blockIdx_y() * (params.output_dim / TILE_N) + thread::blockIdx_x();
                let chunk = cta * WARPS_PER_BLOCK + thread_id;
                unsafe {
                    *output_chunk_amax.get_unchecked_mut(chunk as usize) = 0.0;
                }
            }
            return;
        }
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

        let mut local_amax = 0.0;
        run_csf_relu2_backward_f16_amax!(
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
            pre_activation,
            &mut out,
            &mut local_amax,
        );

        if thread_id < MMA_THREADS_PER_BLOCK {
            let warp_amax = warp_max_nonnegative_f32(local_amax);
            if thread_id & 31 == 0 {
                let cta =
                    thread::blockIdx_y() * (params.output_dim / TILE_N) + thread::blockIdx_x();
                let chunk = cta * WARPS_PER_BLOCK + thread_id / 32;
                unsafe {
                    *output_chunk_amax.get_unchecked_mut(chunk as usize) = warp_amax;
                }
            }
        }
    }
}
