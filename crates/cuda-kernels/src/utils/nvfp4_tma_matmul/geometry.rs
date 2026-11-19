// Shared TMA/MMA implementation, instantiated with each family's geometry.
use cuda_device::{
    DisjointSlice, SharedArray,
    atomic::{AtomicOrdering, DeviceAtomicU32},
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
        mbarrier_init, mbarrier_try_wait_parity,
    },
    convert::cvt_f16x2_f32,
    cuda_module, kernel, launch_bounds, ptx_asm, thread,
    tma::TmaDescriptor,
};

use crate::f16_tc_matmul::convert::{
    cvt_f32_f16, load_f16x2_global_bits_read_only, load_f32_global_read_only,
    load_f32x2_global_read_only, load_f32x2_shared, store_f32x2_shared,
};
use crate::float_ptx::{abs_f32, fma_f32, max_f32, sqrt_f32};
use crate::nvfp4::nvfp4_vector_values2 as nvfp4_values2;
use crate::{
    block_reduce::{block_max_leader_f32, block_sum_shared_f32},
    warp_reduce::{eighth_warp_sum_f32, thread_lane_warp, warp_max_nonnegative_f32},
};

use crate::nvfp4_tma_matmul::cute::{
    KMajorU4, N_LAYOUT_WARP_CONTIGUOUS, PositionIndependentSwizzle64B, Sm120KMajorSwizzle,
    Sm120Nvfp4MmaAtom, Sm120Nvfp4WarpMma, Sm120ScaleLayout, U4SmemSwizzle,
};
use crate::nvfp4_tma_matmul::gradient::{GemmGradientTransform, GradientScalars, GradientSums, store_sums};
use crate::block_reduce::block_sum_quad_leader_f32_for_warps;

const MMA_M: u32 = Sm120Nvfp4MmaAtom::M;
const MMA_N: u32 = Sm120Nvfp4MmaAtom::N;
const MMA_K: u32 = Sm120Nvfp4MmaAtom::K;
const SCALE_MN_BLOCK: u32 = Sm120ScaleLayout::MN_BLOCK;

pub const REQUESTED_TILE_M: u32 = NVFP4_REQUESTED_TILE_M;
pub const REQUESTED_TILE_N: u32 = NVFP4_REQUESTED_TILE_N;
pub const TILE_K: u32 = NVFP4_TILE_K;
const K_ATOMS: u32 = TILE_K / MMA_K;

macro_rules! nvfp4_one_tt_u32 {
    ($item:tt) => {
        1u32
    };
}

macro_rules! nvfp4_count_tts_u32 {
    ($($item:tt),+ $(,)?) => {
        0u32 $(+ nvfp4_one_tt_u32!($item))*
    };
}

macro_rules! define_warp_repeat_consts {
    ([$($m_repeat:tt),+], [$($n_repeat:tt),+]) => {
        const M_REPEAT: u32 = nvfp4_count_tts_u32!($($m_repeat),+);
        const N_REPEAT: u32 = nvfp4_count_tts_u32!($($n_repeat),+);
    };
}

nvfp4_warp_repeat_shape!(define_warp_repeat_consts);
type WarpMmaLayout = Sm120Nvfp4WarpMma<M_REPEAT, N_REPEAT, NVFP4_N_LAYOUT>;
const ACC_COUNT: usize = WarpMmaLayout::ACC_COUNT;
const M_PER_WARP: u32 = WarpMmaLayout::M_PER_WARP;
const N_PER_WARP: u32 = WarpMmaLayout::N_PER_WARP;

const fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        ((value - 1) / divisor) + 1
    }
}

const WARP_TILES_M: u32 = ceil_div_u32(REQUESTED_TILE_M, M_PER_WARP);
const WARP_TILES_N: u32 = ceil_div_u32(REQUESTED_TILE_N, N_PER_WARP);
type WarpTileShape = crate::nvfp4_cute_shape!((WARP_TILES_M, WARP_TILES_N));
pub const TILE_M: u32 = WARP_TILES_M * M_PER_WARP;
pub const TILE_N: u32 = WARP_TILES_N * N_PER_WARP;
const WARPS_PER_BLOCK: u32 = WARP_TILES_M * WARP_TILES_N;
const MMA_THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32;
const POLAR_ROW_SUMS_PER_THREAD: usize = (M_REPEAT * 2) as usize;
const POLAR_ROW_SUMS_PER_CTA: usize = (TILE_M * WARP_TILES_N) as usize;
const TMA_PIPELINE_STAGES: u32 = NVFP4_TMA_STAGES;
const TMA_PIPELINE_STAGES_USIZE: usize = TMA_PIPELINE_STAGES as usize;
const MAX_STATIC_SMEM_BYTES: u32 = 96 * 1024;
pub const TMA_NVFP4_THREADS_PER_BLOCK: u32 = MMA_THREADS_PER_BLOCK + 32;
pub const TMA_NVFP4_RESIDUAL_THREADS_PER_BLOCK: u32 = MMA_THREADS_PER_BLOCK;
const TMA_NVFP4_WARPS_PER_BLOCK: usize = (TMA_NVFP4_THREADS_PER_BLOCK / 32) as usize;

#[inline(always)]
fn atomic_max_nonnegative_f32(output_amax: &[f32], value: f32) {
    let output = unsafe { &*(output_amax.as_ptr() as *const DeviceAtomicU32) };
    output.fetch_max(value.to_bits(), AtomicOrdering::Relaxed);
}

#[inline]
pub const fn tma_nvfp4_output_amax_chunks(token_count: u32, output_dim: u32) -> u32 {
    ceil_div_u32(token_count, TILE_M) * ceil_div_u32(output_dim, TILE_N) * WARPS_PER_BLOCK
}

#[inline]
pub const fn tma_nvfp4_symmetric_output_amax_chunks(dim: u32) -> u32 {
    let tiles = ceil_div_u32(dim, TILE_M);
    (tiles * (tiles + 1) / 2) * (TILE_M / TILE_N) * WARPS_PER_BLOCK
}

const TMA_NVFP4_MAINLOOP_BYTES: u32 =
    A_TILE_BYTES + B_TILE_BYTES + A_SCALE_TILE_BYTES + B_SCALE_TILE_BYTES;

const _: [(); TILE_K as usize] = [(); (K_ATOMS * MMA_K) as usize];
const _: [(); TILE_M as usize] = [(); (WARP_TILES_M * M_PER_WARP) as usize];
const _: [(); TILE_N as usize] = [(); (WARP_TILES_N * N_PER_WARP) as usize];
const _: [(); MMA_THREADS_PER_BLOCK as usize] = [(); (WARPS_PER_BLOCK * 32) as usize];
const _: [(); TMA_NVFP4_THREADS_PER_BLOCK as usize] = [(); (WARPS_PER_BLOCK * 32 + 32) as usize];
const _: [(); ACC_COUNT] = [(); (M_REPEAT * N_REPEAT) as usize];

const PACKS_PER_ROW: u32 = TILE_K / 8;
const A_PACKS: usize = (TILE_M * PACKS_PER_ROW) as usize;
const B_PACKS: usize = (TILE_N * PACKS_PER_ROW) as usize;
const A_SCALE_MN: u32 = Sm120ScaleLayout::padded_mn_extent(TILE_M);
const B_SCALE_MN: u32 = Sm120ScaleLayout::padded_mn_extent(TILE_N);
const A_SCALES: usize = (A_SCALE_MN * K_ATOMS) as usize;
const B_SCALES: usize = (B_SCALE_MN * K_ATOMS) as usize;
const A_TILE_BYTES: u32 = (A_PACKS * core::mem::size_of::<u32>()) as u32;
const B_TILE_BYTES: u32 = (B_PACKS * core::mem::size_of::<u32>()) as u32;
const A_SCALE_TILE_BYTES: u32 = ((A_SCALES) * core::mem::size_of::<u32>()) as u32;
const B_SCALE_TILE_BYTES: u32 = ((B_SCALES) * core::mem::size_of::<u32>()) as u32;
const TMA_NVFP4_STATIC_SMEM_BYTES: u32 = TMA_PIPELINE_STAGES * TMA_NVFP4_MAINLOOP_BYTES;
const A_PACKS_STAGED: usize = A_PACKS * TMA_PIPELINE_STAGES_USIZE;
const B_PACKS_STAGED: usize = B_PACKS * TMA_PIPELINE_STAGES_USIZE;

type APacksSmemStages = SharedArray<u32, A_PACKS_STAGED, 1024>;
type BPacksSmemStages = SharedArray<u32, B_PACKS_STAGED, 1024>;
type NativeAScalesSmemStages = SharedArray<u32, { (A_SCALES) * TMA_PIPELINE_STAGES_USIZE }, 512>;
type NativeBScalesSmemStages = SharedArray<u32, { (B_SCALES) * TMA_PIPELINE_STAGES_USIZE }, 512>;
type CsfAScalesSmemStages = SharedArray<u32, { 288 * TMA_PIPELINE_STAGES_USIZE }, 512>;
type CsfBScalesSmemStages = SharedArray<u32, { 288 * TMA_PIPELINE_STAGES_USIZE }, 512>;
type BarrierSmemStages = SharedArray<Barrier, TMA_PIPELINE_STAGES_USIZE, 8>;
type ResidualAPacksStages = SharedArray<u32, { A_PACKS * 2 }, 1024>;
type ResidualBPacksStages = SharedArray<u32, { B_PACKS * 2 }, 1024>;
type ResidualScalesStages = SharedArray<u32, { 288 * 2 }, 512>;
type ResidualBarrierStages = SharedArray<Barrier, 2, 8>;
type BiasSmem = SharedArray<f32, { TILE_N as usize }, 128>;
type TmaOperandLayout = KMajorU4<PACKS_PER_ROW, Sm120KMajorSwizzle<PACKS_PER_ROW>>;

#[inline(always)]
fn prefetch_tma_descriptor(desc: *const TmaDescriptor) {
    unsafe {
        ptx_asm!("prefetch.tensormap [%0];", in("l") desc as u64);
    }
}

#[inline(always)]
fn shared_addr_u32<T>(ptr: *mut T) -> u32 {
    let shared_addr: u32;
    unsafe {
        ptx_asm!(
            r"{
              .reg .u64 smem64;
              cvta.to.shared.u64 smem64, %1;
              cvt.u32.u64 %0, smem64;
            }",
            out("=r") shared_addr,
            in("l") ptr as u64,
            options(register_only),
        );
    }
    shared_addr
}

#[inline(always)]
fn cp_async_bulk_tensor_2d_cta_g2s(
    dst: *mut u8,
    desc: *const TmaDescriptor,
    x: i32,
    y: i32,
    barrier: *mut Barrier,
) {
    let dst_smem = shared_addr_u32(dst);
    let barrier_smem = shared_addr_u32(barrier);
    let cache_hint = 0u64;
    unsafe {
        ptx_asm!(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;",
            in("r") dst_smem,
            in("l") desc as u64,
            in("r") barrier_smem,
            in("r") x,
            in("r") y,
            in("l") cache_hint,
            clobber("memory"),
        );
    }
}

#[inline(always)]
fn cp_async_csf_codes(dst: *mut u8, desc: *const TmaDescriptor, tile: u32, barrier: *mut Barrier) {
    unsafe {
        ptx_asm!("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {0, 0, %3}], [%2];",
        in("r")shared_addr_u32(dst),in("l")desc as u64,in("r")shared_addr_u32(barrier),in("r")tile,clobber("memory"));
    }
}

#[inline(always)]
fn wait_mbarrier_parity(bar: *mut Barrier, parity: u32) {
    while unsafe { !mbarrier_try_wait_parity(bar as *const Barrier, parity) } {}
}

#[inline(always)]
fn stage_ptr<T>(base: *mut T, stage: u32, elems_per_stage: usize) -> *mut T {
    unsafe { base.add(stage as usize * elems_per_stage) }
}

#[inline(always)]
fn stage_barrier(base: *mut Barrier, stage: u32) -> *mut Barrier {
    unsafe { base.add(stage as usize) }
}

#[derive(Clone, Copy)]
struct CtaTile {
    row_base: u32,
    col_base: u32,
    warp_m: u32,
    warp_n: u32,
    n_layout: u32,
    group: u32,
    thread_in_group: u32,
}

impl CtaTile {
    #[inline(always)]
    fn new(thread_id: u32) -> Self {
        Self::new_with_n_layout(thread_id, NVFP4_N_LAYOUT)
    }

    #[inline(always)]
    fn new_csf(thread_id: u32) -> Self {
        Self::new_with_n_layout(thread_id, N_LAYOUT_WARP_CONTIGUOUS)
    }

    #[inline(always)]
    fn new_with_n_layout(thread_id: u32, n_layout: u32) -> Self {
        Self::new_at(
            thread_id,
            n_layout,
            thread::blockIdx_y(),
            thread::blockIdx_x(),
        )
    }

    #[inline(always)]
    fn new_at(thread_id: u32, n_layout: u32, tile_row: u32, tile_col: u32) -> Self {
        let lane = thread_id & 31;
        let warp = thread_id >> 5;
        let (warp_m, warp_n) = WarpTileShape::unflatten(warp);
        Self {
            row_base: tile_row * TILE_M,
            col_base: tile_col * TILE_N,
            warp_m,
            warp_n,
            n_layout,
            group: lane >> 2,
            thread_in_group: lane & 3,
        }
    }

    #[inline(always)]
    fn mma_row_base(self, m_repeat: u32) -> u32 {
        self.row_base + self.warp_m * M_PER_WARP + m_repeat * MMA_M
    }

    #[inline(always)]
    fn mma_col_base(self, n_repeat: u32) -> u32 {
        self.col_base + self.mma_col_offset(n_repeat)
    }

    #[inline(always)]
    fn mma_col_offset(self, n_repeat: u32) -> u32 {
        mma_n_atom(self.n_layout, self.warp_n, n_repeat) * MMA_N
    }
}

#[inline(always)]
fn route_mask(params: Nvfp4GemmParams, index: u32) -> u64 {
    unsafe { *((params.route_masks as *const u64).add(index as usize)) }
}

#[inline(always)]
fn token_feature_mask(tile: CtaTile, params: Nvfp4GemmParams) -> u64 {
    route_mask(params, tile.row_base / TILE_M)
}

#[derive(Clone,Copy)]
struct RouteBits {low:u64,high:u64}
impl RouteBits {
    #[inline(always)] fn any(self)->bool {self.low!=0||self.high!=0}
    #[inline(always)] fn count(self)->u32 {self.low.count_ones()+self.high.count_ones()}
    #[inline(always)] fn take_first(&mut self)->u32 {
        if self.low!=0 {
            let bit=self.low.trailing_zeros();self.low&=self.low.wrapping_sub(1);bit
        }else{
            let bit=self.high.trailing_zeros();self.high&=self.high.wrapping_sub(1);64+bit
        }
    }
}
#[inline(always)]
fn feature_token_mask(feature_tile:u32,params:Nvfp4GemmParams)->RouteBits {
    let words=params.route_token_tiles.div_ceil(64);
    let base=params.route_token_tiles+feature_tile*words;
    RouteBits {low:route_mask(params,base),high:if words>1{route_mask(params,base+1)}else{0}}
}

#[inline(always)]
fn routed_output_tile_active(tile: CtaTile, params: Nvfp4GemmParams) -> bool {
    match params.route_mode {
        ROUTE_TOKEN_M_FEATURE_N => {
            let feature_tile = tile.col_base / SCALE_MN_BLOCK;
            token_feature_mask(tile, params) & (1_u64 << feature_tile) != 0
        }
        ROUTE_FEATURE_M_TOKEN_K => feature_token_mask(tile.row_base / TILE_M, params).any(),
        ROUTE_FEATURE_N_TOKEN_K => feature_token_mask(tile.col_base / SCALE_MN_BLOCK, params).any(),
        _ => true,
    }
}

#[inline(always)]
fn routed_k_tile_count(tile: CtaTile, params: Nvfp4GemmParams) -> u32 {
    match params.route_mode {
        ROUTE_TOKEN_M_FEATURE_K => params.route_active_tiles,
        ROUTE_FEATURE_M_TOKEN_K => feature_token_mask(tile.row_base / TILE_M, params).count(),
        ROUTE_FEATURE_N_TOKEN_K => {
            feature_token_mask(tile.col_base / SCALE_MN_BLOCK, params).count()
        }
        _ => params.input_dim / TILE_K,
    }
}

#[inline(always)]
fn routed_k_mask(tile: CtaTile, params: Nvfp4GemmParams) -> RouteBits {
    match params.route_mode {
        ROUTE_TOKEN_M_FEATURE_K => RouteBits{low:token_feature_mask(tile,params),high:0},
        ROUTE_FEATURE_M_TOKEN_K => feature_token_mask(tile.row_base / TILE_M, params),
        ROUTE_FEATURE_N_TOKEN_K => feature_token_mask(tile.col_base / SCALE_MN_BLOCK, params),
        _ => RouteBits{low:0,high:0},
    }
}

#[inline(always)]
fn zero_output_tile_f32(tile: CtaTile, params: Nvfp4GemmParams, out: &mut DisjointSlice<f32>) {
    let thread_id = thread::threadIdx_x();
    let thread_count = thread::blockDim_x();
    let mut local = thread_id;
    while local < TILE_M * TILE_N {
        let row = tile.row_base + local / TILE_N;
        let col = tile.col_base + local % TILE_N;
        unsafe {
            *out.get_unchecked_mut((row * params.output_dim + col) as usize) = 0.0;
        }
        local += thread_count;
    }
}

#[inline(always)]
const fn mma_n_atom(n_layout: u32, warp_n: u32, n_repeat: u32) -> u32 {
    if n_layout == N_LAYOUT_WARP_CONTIGUOUS {
        Sm120Nvfp4WarpMma::<M_REPEAT, N_REPEAT, N_LAYOUT_WARP_CONTIGUOUS>::n_atom::<WARP_TILES_N>(
            warp_n, n_repeat,
        )
    } else {
        WarpMmaLayout::n_atom::<WARP_TILES_N>(warp_n, n_repeat)
    }
}

#[inline(always)]
const fn mma_n_atoms_are_adjacent(n_layout: u32, warp_n: u32, n0: u32, n1: u32) -> bool {
    mma_n_atom(n_layout, warp_n, n1) == mma_n_atom(n_layout, warp_n, n0) + 1
}

#[inline(always)]
fn mma_m16n8k64_mxf4nvf4_scale4x_ue4m3(
    a: [u32; 4],
    b: [u32; 2],
    acc: &mut [f32; 4],
    scale_a: ScaleWord,
    scale_b: ScaleWord,
) {
    unsafe {
        let packed: u128;
        ptx_asm!(
            r"{
             .reg .b32 %%d0, %%d1, %%d2, %%d3;
             mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
             {%%d0, %%d1, %%d2, %%d3},
             {%1, %2, %3, %4},
             {%5, %6},
             {%7, %8, %9, %10},
             %11, {0, %13}, %12, {0, %14};
             mov.b128 %0, {%%d0, %%d1, %%d2, %%d3};
             }",
            out("=q") packed,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b[0],
            in("r") b[1],
            in("f") acc[0],
            in("f") acc[1],
            in("f") acc[2],
            in("f") acc[3],
            in("r") scale_a.data,
            in("r") scale_b.data,
            in("h") scale_a.selector,
            in("h") scale_b.selector,
            options(register_only),
        );
        acc[0] = f32::from_bits(packed as u32);
        acc[1] = f32::from_bits((packed >> 32) as u32);
        acc[2] = f32::from_bits((packed >> 64) as u32);
        acc[3] = f32::from_bits((packed >> 96) as u32);
    }
}

#[inline(always)]
fn store_f32x2_global(out: &mut DisjointSlice<f32>, index: u32, x: f32, y: f32) {
    unsafe {
        let ptr = out.get_unchecked_mut(index as usize) as *mut f32;
        ptx_asm!("st.global.v2.f32 [%0], {%1, %2};",in("l")ptr as u64,in("f")x,in("f")y);
    }
}

#[inline(always)]
fn store_f16x2_global(out: *mut u16, index: u32, x: f32, y: f32) {
    let packed = cvt_f16x2_f32(x, y);
    unsafe {
        ptx_asm!(
            "st.global.u32 [%0], %1;",
            in("l") out.add(index as usize) as u64,
            in("r") packed,
        );
    }
}

#[derive(Clone, Copy)]
struct OutputScale {
    mode: u32,
    weight: f32,
    scalar_a: f32,
    scalar_b: f32,
    a_global_scale: u64,
}

impl OutputScale {
    #[inline(always)]
    fn new(params: Nvfp4GemmParams) -> Self {
        let (scalar_a, scalar_b) = if params.global_scale_mode == 0 {
            (1.0, 1.0)
        } else {
            let scalar_b =
                load_f32_global_read_only(params.b_global_scale as usize as *const f32, 0);
            let scalar_a = if params.global_scale_mode == 1 {
                load_f32_global_read_only(params.a_global_scale as usize as *const f32, 0)
            } else {
                1.0
            };
            (scalar_a, scalar_b)
        };
        Self {
            mode: params.global_scale_mode,
            weight: params.weight_global_scale,
            scalar_a,
            scalar_b,
            a_global_scale: params.a_global_scale,
        }
    }

    #[inline(always)]
    fn row(self, row: u32) -> f32 {
        if self.mode == 2 {
            let row_scale =
                load_f32_global_read_only(self.a_global_scale as usize as *const f32, row as usize);
            self.weight * row_scale * self.scalar_b
        } else {
            self.weight * self.scalar_a * self.scalar_b
        }
    }
}

#[inline(always)]
fn scale_tma_y(mn_base: u32, k_base: u32, mn_extent: u32, k_dim: u32) -> i32 {
    Sm120ScaleLayout::block_major_tma_y_padded(mn_base, k_base, mn_extent, k_dim)
        / Sm120ScaleLayout::TMA_SWIZZLE_ROW_FACTOR as i32
}

trait ScaleFragment: Copy {
    fn load(self, row: u32, k_atom: u32, mn_extent: u32, active: bool, _repeat: u32) -> u32;
    fn selector(self, _row: u32, _k: u32, _repeat: u32) -> u16 {
        0
    }
}
#[derive(Clone, Copy)]
struct ScaleWord {
    data: u32,
    selector: u16,
}

#[derive(Clone, Copy)]
struct HardwareScaleFragment<const A: bool> {
    lo: u128,
    hi: u128,
    first_half: u32,
}
impl<const A: bool> ScaleFragment for HardwareScaleFragment<A> {
    #[inline(always)]
    fn load(self, _row: u32, _k: u32, _mn: u32, _active: bool, repeat: u32) -> u32 {
        let at = if A {
            repeat
        } else {
            (repeat + self.first_half) / 2
        };
        let packed = if at < 4 { self.lo } else { self.hi };
        (packed >> ((at & 3) * 32)) as u32
    }
    #[inline(always)]
    fn selector(self, _row: u32, k: u32, repeat: u32) -> u16 {
        if A {
            k as u16
        } else {
            (((repeat + self.first_half) & 1) + 2 * k) as u16
        }
    }
}
// Retain both K halves and adjacent N halves in one register word. This
// uses the same MMA scale-fragment layout as the compressed matrix loads.
#[inline(always)]
fn load_native_scale_word(shared: *const u32, first: u32, extent: u32) -> u32 {
    let lane = cuda_device::warp::lane_id();
    let row = first + lane / 4 + (lane & 1) * 8;
    unsafe { *shared.add(scale_packed_u32_index(row, (lane >> 1) & 1, extent)) }
}
#[inline(always)]
fn load_native_scales<const COUNT: usize, const A: bool>(
    shared: *const u32,
    tile: CtaTile,
) -> HardwareScaleFragment<A> {
    macro_rules! value {
        ($i:expr) => {{
            if COUNT > $i {
                let row = if A {
                    tile.row_base % 128 + tile.warp_m * M_PER_WARP + $i * MMA_M
                } else {
                    tile.col_base % 128 + (tile.mma_col_offset($i * 2) & !15)
                };
                u128::from(load_native_scale_word(
                    shared,
                    row,
                    if A { A_SCALE_MN } else { B_SCALE_MN },
                ))
            } else {
                0
            }
        }};
    }
    let lo = value!(0) | (value!(1) << 32) | (value!(2) << 64) | (value!(3) << 96);
    let hi = value!(4) | (value!(5) << 32) | (value!(6) << 64) | (value!(7) << 96);
    HardwareScaleFragment {
        lo,
        hi,
        first_half: if A {
            0
        } else {
            (tile.mma_col_offset(0) >> 3) & 1
        },
    }
}
/// Hardware unpacks low six bits in MMA order. One integer multiply deposits
/// the four high bits directly into their final byte positions.
#[inline(always)]
fn load_scale_matrix_codes<const MATRICES: u32>(address: u32) -> u128 {
    if MATRICES == 4 {
        let v: u128;
        unsafe {
            ptx_asm!(r"{
        .reg .b32 a,b,c,d;
        ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {a,b,c,d},[%1];
        mov.b128 %0,{a,b,c,d};
    }",out("=q")v,in("r")address,clobber("memory"));
        }
        v
    } else if MATRICES == 2 {
        let v: u64;
        unsafe {
            ptx_asm!(r"{
        .reg .b32 a,b;
        ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 {a,b},[%1];
        mov.b64 %0,{a,b};
    }",out("=l")v,in("r")address,clobber("memory"));
        }
        u128::from(v)
    } else {
        let v: u32;
        unsafe {
            ptx_asm!("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%0},[%1];",out("=r")v,in("r")address,clobber("memory"));
        }
        u128::from(v)
    }
}
#[derive(Clone, Copy)]
struct RawScaleBatch {
    lo: u128,
    hi: u128,
    mask_lo: u32,
    mask_hi: u32,
}
#[inline(always)]
fn load_scale_masks<const COUNT: usize, const STEP: u32>(sm: u32, first: u32) -> u32 {
    let lane = cuda_device::warp::lane_id();
    let row = (first & !15) + lane / 4 + (lane & 1) * 8;
    let at = sm + 1024 + row;
    let a: u32;
    unsafe {
        ptx_asm!("ld.shared.u8 %0,[%1];",out("=r")a,in("r")at,clobber("memory"));
    }
    let b = if COUNT > 1 {
        let v: u32;
        unsafe {
            ptx_asm!("ld.shared.u8 %0,[%1];",out("=r")v,in("r")at+STEP*16,clobber("memory"));
        }
        v
    } else {
        0
    };
    let c = if COUNT > 2 {
        let v: u32;
        unsafe {
            ptx_asm!("ld.shared.u8 %0,[%1];",out("=r")v,in("r")at+STEP*32,clobber("memory"));
        }
        v
    } else {
        0
    };
    let d = if COUNT > 3 {
        let v: u32;
        unsafe {
            ptx_asm!("ld.shared.u8 %0,[%1];",out("=r")v,in("r")at+STEP*48,clobber("memory"));
        }
        v
    } else {
        0
    };
    a | (b << 8) | (c << 16) | (d << 24)
}
#[inline(always)]
fn load_hardware_scales_raw<const COUNT: usize, const STEP: u32>(
    shared: *mut u32,
    first_row: u32,
) -> RawScaleBatch {
    let lane = cuda_device::warp::lane_id();
    let first = first_row & !15;
    let sm = shared_addr_u32(shared);
    let width = if COUNT > 2 {
        32
    } else if COUNT > 1 {
        16
    } else {
        8
    };
    let source = lane % width;
    let address = sm + (first / 16 * 8 + (source / 8) * 8 * STEP + source % 8) * 16;
    let lo = if COUNT > 2 {
        load_scale_matrix_codes::<4>(address)
    } else if COUNT > 1 {
        load_scale_matrix_codes::<2>(address)
    } else {
        load_scale_matrix_codes::<1>(address)
    };
    let hi = if COUNT > 4 {
        load_scale_matrix_codes::<4>(
            sm + ((first + 64 * STEP) / 16 * 8 + (lane / 8) * 8 * STEP + lane % 8) * 16,
        )
    } else {
        0
    };
    RawScaleBatch {
        lo,
        hi,
        mask_lo: load_scale_masks::<COUNT, STEP>(sm, first),
        mask_hi: if COUNT > 4 {
            load_scale_masks::<4, STEP>(sm, first + 64 * STEP)
        } else {
            0
        },
    }
}
#[inline(always)]
fn scale_raw_at(batch: RawScaleBatch, index: u32) -> u64 {
    let (code, bases) = if index < 4 {
        (batch.lo, batch.mask_lo)
    } else {
        (batch.hi, batch.mask_hi)
    };
    u64::from((code >> ((index & 3) * 32)) as u32)
        | (u64::from((bases >> ((index & 3) * 8)) & 255) << 32)
}
#[inline(always)]
fn decode_hardware_scale(raw: u64) -> u32 {
    let half = (cuda_device::warp::lane_id() >> 1) & 1;
    let mask = ((raw >> 32) as u32 >> (half * 4)) & 15;
    (raw as u32) | ((mask.wrapping_mul(0x08102040)) & 0x40404040)
}

#[inline(always)]
fn decode_hardware_scales<const COUNT: usize, const A: bool>(
    raw: RawScaleBatch,
    half: u32,
) -> HardwareScaleFragment<A> {
    macro_rules! value {
        ($i:expr) => {{
            if COUNT > $i {
                u128::from(decode_hardware_scale(scale_raw_at(raw, $i)))
            } else {
                0
            }
        }};
    }
    let lo = value!(0) | (value!(1) << 32) | (value!(2) << 64) | (value!(3) << 96);
    let hi = value!(4) | (value!(5) << 32) | (value!(6) << 64) | (value!(7) << 96);
    HardwareScaleFragment {
        lo,
        hi,
        first_half: half,
    }
}

#[inline(always)]
fn stage_csf_tile(
    a_tma: *const TmaDescriptor,
    b_tma: *const TmaDescriptor,
    a_scale_tma: *const TmaDescriptor,
    b_scale_tma: *const TmaDescriptor,
    tile: CtaTile,
    k_base: u32,
    params: Nvfp4GemmParams,
    a_packs: *mut u32,
    b_packs: *mut u32,
    a_scales: *mut u32,
    b_scales: *mut u32,
    bar: *mut Barrier,
) {
    let at = tile.row_base / 128 * (params.input_dim / 128) + k_base / 128;
    let bt = tile.col_base / 128 * (params.input_dim / 128) + k_base / 128;
    unsafe {
        mbarrier_arrive_expect_tx(bar as *const Barrier, 1, A_TILE_BYTES + B_TILE_BYTES + 1792);
        cp_async_bulk_tensor_2d_cta_g2s(
            a_packs as *mut u8,
            a_tma,
            k_base as i32,
            tile.row_base as i32,
            bar,
        );
        cp_async_bulk_tensor_2d_cta_g2s(
            b_packs as *mut u8,
            b_tma,
            k_base as i32,
            tile.col_base as i32,
            bar,
        );
        cp_async_csf_codes(a_scales as *mut u8, a_scale_tma, at, bar);
        cp_async_csf_codes(b_scales as *mut u8, b_scale_tma, bt, bar);
        cp_async_csf_codes(a_scales.add(256) as *mut u8, a_scale_tma.add(1), at, bar);
        cp_async_csf_codes(b_scales.add(256) as *mut u8, b_scale_tma.add(1), bt, bar);
    }
}

#[inline(always)]
fn stage_tiles_full_tma_nvfp4(
    a_tma: *const TmaDescriptor,
    b_tma: *const TmaDescriptor,
    a_scale_tma: *const TmaDescriptor,
    b_scale_tma: *const TmaDescriptor,
    tile: CtaTile,
    k_base: u32,
    params: Nvfp4GemmParams,
    a_packs: *mut u32,
    b_packs: *mut u32,
    a_scales: *mut u32,
    b_scales: *mut u32,
    tma_bar: *mut Barrier,
) {
    if thread::threadIdx_x() == MMA_THREADS_PER_BLOCK {
        unsafe {
            mbarrier_arrive_expect_tx(
                tma_bar as *const Barrier,
                1,
                A_TILE_BYTES + B_TILE_BYTES + A_SCALE_TILE_BYTES + B_SCALE_TILE_BYTES,
            );
            cp_async_bulk_tensor_2d_cta_g2s(
                a_packs as *mut u8,
                a_tma,
                k_base as i32,
                tile.row_base as i32,
                tma_bar,
            );
            cp_async_bulk_tensor_2d_cta_g2s(
                b_packs as *mut u8,
                b_tma,
                k_base as i32,
                tile.col_base as i32,
                tma_bar,
            );
            cp_async_bulk_tensor_2d_cta_g2s(
                a_scales as *mut u8,
                a_scale_tma,
                0,
                scale_tma_y(tile.row_base, k_base, params.token_count, params.input_dim),
                tma_bar,
            );
            cp_async_bulk_tensor_2d_cta_g2s(
                b_scales as *mut u8,
                b_scale_tma,
                0,
                scale_tma_y(tile.col_base, k_base, params.output_dim, params.input_dim),
                tma_bar,
            );
        }
    }
}

#[inline(always)]
fn ldmatrix_m8n8_x4_shared_b16(ptr: *const u32) -> [u32; 4] {
    let packed: u128;
    unsafe {
        ptx_asm!(
            r"{
             .reg .b32 %%r0, %%r1, %%r2, %%r3;
             .reg .u64 %%smem64;
             .reg .u32 %%smem32;
             cvta.to.shared.u64 %%smem64, %1;
             cvt.u32.u64 %%smem32, %%smem64;
             ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%%r0, %%r1, %%r2, %%r3}, [%%smem32];
             mov.b128 %0, {%%r0, %%r1, %%r2, %%r3};
             }",
            out("=q") packed,
            in("l") ptr as u64,
            clobber("memory"),
        );
    }
    [
        packed as u32,
        (packed >> 32) as u32,
        (packed >> 64) as u32,
        (packed >> 96) as u32,
    ]
}

#[inline(always)]
fn load_a_fragments_swizzled(
    a_packs: *const u32,
    tile: CtaTile,
    k_atom: u32,
    m_repeat: u32,
) -> [u32; 4] {
    let lane = tile.group * 4 + tile.thread_in_group;
    let matrix = lane >> 3;
    let row_in_matrix = lane & 7;
    let row = tile.warp_m * M_PER_WARP
        + m_repeat * MMA_M
        + row_in_matrix
        + if matrix & 1 == 0 { 0 } else { MMA_M / 2 };
    let pack = k_atom * (MMA_K / 8) + if matrix < 2 { 0 } else { 4 };
    let ptr = unsafe { a_packs.add(TmaOperandLayout::ldmatrix_chunk_start_u32(row, pack)) };
    ldmatrix_m8n8_x4_shared_b16(ptr)
}

#[inline(always)]
#[allow(dead_code)]
fn tma_operand_pack_index(row: u32, pack: u32) -> usize {
    TmaOperandLayout::u32_index(row, pack)
}

#[inline(always)]
#[allow(dead_code)]
fn load_b_fragment_swizzled(
    b_packs: *const u32,
    tile: CtaTile,
    k_atom: u32,
    n_repeat: u32,
    register: u32,
) -> u32 {
    let col = tile.mma_col_offset(n_repeat) + tile.group;
    let pack = k_atom * (MMA_K / 8) + tile.thread_in_group + if register == 0 { 0 } else { 4 };
    unsafe { *b_packs.add(tma_operand_pack_index(col, pack)) }
}

#[inline(always)]
#[allow(dead_code)]
fn load_b_fragments_swizzled(
    b_packs: *const u32,
    tile: CtaTile,
    k_atom: u32,
    n_repeat: u32,
) -> [u32; 2] {
    [
        load_b_fragment_swizzled(b_packs, tile, k_atom, n_repeat, 0),
        load_b_fragment_swizzled(b_packs, tile, k_atom, n_repeat, 1),
    ]
}

#[inline(always)]
fn load_b_fragment_pair_ldmatrix(
    b_packs: *const u32,
    tile: CtaTile,
    k_atom: u32,
    n_repeat: u32,
) -> [[u32; 2]; 2] {
    let lane = tile.group * 4 + tile.thread_in_group;
    let matrix = lane >> 3;
    let n_atom = mma_n_atom(tile.n_layout, tile.warp_n, n_repeat) + (matrix >> 1);
    let col = n_atom * MMA_N + (lane & (MMA_N - 1));
    let pack = k_atom * (MMA_K / 8) + if matrix & 1 == 0 { 0 } else { 4 };
    let ptr = unsafe { b_packs.add(TmaOperandLayout::ldmatrix_chunk_start_u32(col, pack)) };
    let regs = ldmatrix_m8n8_x4_shared_b16(ptr);
    [[regs[0], regs[1]], [regs[2], regs[3]]]
}

#[inline(always)]
fn scale_packed_u32_index(mn: u32, k_atom: u32, mn_extent: u32) -> usize {
    let logical = Sm120ScaleLayout::u32_word_offset(mn, k_atom, mn_extent) as u32;
    PositionIndependentSwizzle64B::u32_index(logical) as usize
}

#[inline(always)]
fn scale_a_row(tile: CtaTile, m_repeat: u32) -> u32 {
    (tile.row_base % SCALE_MN_BLOCK)
        + tile.warp_m * M_PER_WARP
        + m_repeat * MMA_M
        + tile.group
        + if tile.thread_in_group & 1 == 1 {
            MMA_M / 2
        } else {
            0
        }
}
#[inline(always)]
fn scale_b_row(tile: CtaTile, n_repeat: u32) -> u32 {
    (tile.col_base % SCALE_MN_BLOCK) + tile.mma_col_offset(n_repeat) + tile.group
}
#[inline(always)]
fn load_a_scale4_packed(
    a_scales: impl ScaleFragment,
    tile: CtaTile,
    k_atom: u32,
    m_repeat: u32,
) -> ScaleWord {
    let row = scale_a_row(tile, m_repeat);
    ScaleWord {
        data: a_scales.load(row, k_atom, A_SCALE_MN, tile.thread_in_group < 2, m_repeat),
        selector: a_scales.selector(row, k_atom, m_repeat),
    }
}
#[inline(always)]
fn load_b_scale4_packed(
    b_scales: impl ScaleFragment,
    tile: CtaTile,
    k_atom: u32,
    n_repeat: u32,
) -> ScaleWord {
    let row = scale_b_row(tile, n_repeat);
    ScaleWord {
        data: b_scales.load(row, k_atom, B_SCALE_MN, tile.thread_in_group == 0, n_repeat),
        selector: b_scales.selector(row, k_atom, n_repeat),
    }
}

#[inline(always)]
fn store_acc_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    store_f32x2_global(
        out,
        row0 * output_dim + col0,
        acc[0] * scale0,
        acc[1] * scale0,
    );
    store_f32x2_global(
        out,
        row1 * output_dim + col0,
        acc[2] * scale1,
        acc[3] * scale1,
    );
}

#[inline(always)]
fn store_acc_scaled_exact(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;

    store_f32x2_global(
        out,
        row0 * output_dim + col0,
        acc[0] * scale0,
        acc[1] * scale0,
    );
    store_f32x2_global(
        out,
        row1 * output_dim + col0,
        acc[2] * scale1,
        acc[3] * scale1,
    );
}

#[inline(always)]
fn store_acc_scaled_gradient(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
    state: GradientScalars,
    sums: &mut GradientSums,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let index0 = row0 * params.output_dim + col0;
    let index1 = row1 * params.output_dim + col0;
    let (v0, v1) = state.pair(index0, acc[0] * scale0, acc[1] * scale0, sums);
    let (v2, v3) = state.pair(index1, acc[2] * scale1, acc[3] * scale1, sums);
    store_f32x2_global(out, index0, v0, v1);
    store_f32x2_global(out, index1, v2, v3);
}

#[inline(always)]
fn store_acc_scaled_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let value00 = acc[0] * scale0;
    let value01 = acc[1] * scale0;
    let value10 = acc[2] * scale1;
    let value11 = acc[3] * scale1;
    *local_amax = max_f32(
        *local_amax,
        max_f32(
            max_f32(abs_f32(value00), abs_f32(value01)),
            max_f32(abs_f32(value10), abs_f32(value11)),
        ),
    );
    store_f32x2_global(out, row0 * output_dim + col0, value00, value01);
    store_f32x2_global(out, row1 * output_dim + col0, value10, value11);
}

#[inline(always)]
fn store_acc_relu2_backward_f16_scaled_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    pre_activation: &[u16],
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;

    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;
    let pre0 = load_f16x2_global_bits_read_only(pre_activation.as_ptr(), index0 as usize);
    let pre1 = load_f16x2_global_bits_read_only(pre_activation.as_ptr(), index1 as usize);
    let pre00 = cvt_f32_f16(pre0 as u16);
    let pre01 = cvt_f32_f16((pre0 >> 16) as u16);
    let pre10 = cvt_f32_f16(pre1 as u16);
    let pre11 = cvt_f32_f16((pre1 >> 16) as u16);
    let value00 = (acc[0] * scale0) * 2.0 * max_f32(pre00, 0.0);
    let value01 = (acc[1] * scale0) * 2.0 * max_f32(pre01, 0.0);
    let value10 = (acc[2] * scale1) * 2.0 * max_f32(pre10, 0.0);
    let value11 = (acc[3] * scale1) * 2.0 * max_f32(pre11, 0.0);
    *local_amax = max_f32(
        *local_amax,
        max_f32(
            max_f32(abs_f32(value00), abs_f32(value01)),
            max_f32(abs_f32(value10), abs_f32(value11)),
        ),
    );
    store_f32x2_global(out, index0, value00, value01);
    store_f32x2_global(out, index1, value10, value11);
}

include!("polar_epilogues.rs");

#[inline(always)]
fn store_acc_scaled_symmetric_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
    local_amax: &mut f32,
    mirror: bool,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let col1 = col0 + 1;
    let output_dim = params.output_dim;

    let value00 = acc[0] * scale0;
    let value01 = acc[1] * scale0;
    let value10 = acc[2] * scale1;
    let value11 = acc[3] * scale1;
    *local_amax = max_f32(
        *local_amax,
        max_f32(
            max_f32(abs_f32(value00), abs_f32(value01)),
            max_f32(abs_f32(value10), abs_f32(value11)),
        ),
    );
    store_f32x2_global(out, row0 * output_dim + col0, value00, value01);
    store_f32x2_global(out, row1 * output_dim + col0, value10, value11);

    if mirror {
        unsafe {
            *out.get_unchecked_mut((col0 * output_dim + row0) as usize) = value00;
            *out.get_unchecked_mut((col1 * output_dim + row0) as usize) = value01;
            *out.get_unchecked_mut((col0 * output_dim + row1) as usize) = value10;
            *out.get_unchecked_mut((col1 * output_dim + row1) as usize) = value11;
        }
    }
}

#[inline(always)]
fn affine_from_stored_product(acc: f32, scale: f32, bias: f32) -> f32 {
    let value: f32;
    unsafe {
        ptx_asm!(
            "{ .reg .f32 raw; mul.rn.f32 raw, %1, %2; add.rn.f32 %0, raw, %3; }",
            out("=f") value,
            in("f") acc,
            in("f") scale,
            in("f") bias,
            options(register_only),
        );
    }
    value
}

#[inline(always)]
fn store_acc_affine_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    out: &mut DisjointSlice<f32>,
    bias_tile: &BiasSmem,
    output_stride: u32,
    output_offset: u32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let bias_col = tile.mma_col_offset(n_repeat) + tile.thread_in_group * 2;
    let (bias0, bias1) = load_f32x2_shared(bias_tile.as_ptr(), bias_col as usize);
    store_f32x2_global(
        out,
        row0 * output_stride + output_offset + col0,
        affine_from_stored_product(acc[0], scale0, bias0),
        affine_from_stored_product(acc[1], scale0, bias1),
    );
    store_f32x2_global(
        out,
        row1 * output_stride + output_offset + col0,
        affine_from_stored_product(acc[2], scale1, bias0),
        affine_from_stored_product(acc[3], scale1, bias1),
    );
}

#[inline(always)]
fn store_acc_residual_scaled<const ADD_RESIDUAL: bool>(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    residual: &mut DisjointSlice<f32>,
    bias_tile: &BiasSmem,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let index00 = row0 * output_dim + col0;
    let index10 = row1 * output_dim + col0;
    let bias_col = tile.mma_col_offset(n_repeat) + tile.thread_in_group * 2;
    let (bias0, bias1) = load_f32x2_shared(bias_tile.as_ptr(), bias_col as usize);
    let value00 = affine_from_stored_product(acc[0], scale0, bias0);
    let value01 = affine_from_stored_product(acc[1], scale0, bias1);
    let value10 = affine_from_stored_product(acc[2], scale1, bias0);
    let value11 = affine_from_stored_product(acc[3], scale1, bias1);
    let (residual00, residual01, residual10, residual11) = if ADD_RESIDUAL {
        unsafe {
            (
                *residual.get_unchecked_mut(index00 as usize),
                *residual.get_unchecked_mut((index00 + 1) as usize),
                *residual.get_unchecked_mut(index10 as usize),
                *residual.get_unchecked_mut((index10 + 1) as usize),
            )
        }
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    store_f32x2_global(
        residual,
        index00,
        fma_f32(1.0, value00, residual00),
        fma_f32(1.0, value01, residual01),
    );
    store_f32x2_global(
        residual,
        index10,
        fma_f32(1.0, value10, residual10),
        fma_f32(1.0, value11, residual11),
    );
}

#[inline(always)]
fn store_acc_relu2_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    pre_activation: &mut DisjointSlice<f32>,
    out: &mut DisjointSlice<f32>,
    bias_bytes: &[u8],
    bias_scales: &[u8],
    bias_global_scale: f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let (bias0, bias1) = nvfp4_values2(bias_bytes, bias_scales, bias_global_scale, col0 as usize);
    let pre00 = affine_from_stored_product(acc[0], scale0, bias0);
    let pre01 = affine_from_stored_product(acc[1], scale0, bias1);
    let pre10 = affine_from_stored_product(acc[2], scale1, bias0);
    let pre11 = affine_from_stored_product(acc[3], scale1, bias1);
    let relu00 = max_f32(pre00, 0.0);
    let relu01 = max_f32(pre01, 0.0);
    let relu10 = max_f32(pre10, 0.0);
    let relu11 = max_f32(pre11, 0.0);

    store_f32x2_global(pre_activation, row0 * output_dim + col0, pre00, pre01);
    store_f32x2_global(pre_activation, row1 * output_dim + col0, pre10, pre11);
    store_f32x2_global(
        out,
        row0 * output_dim + col0,
        crate::activation::centered_relu2_from_positive(relu00),
        crate::activation::centered_relu2_from_positive(relu01),
    );
    store_f32x2_global(
        out,
        row1 * output_dim + col0,
        crate::activation::centered_relu2_from_positive(relu10),
        crate::activation::centered_relu2_from_positive(relu11),
    );
}

#[inline(always)]
fn store_acc_relu2_f16_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    pre_activation: &mut DisjointSlice<f32>,
    pre_activation_f16: *mut u16,
    out: &mut DisjointSlice<f32>,
    bias_bytes: &[u8],
    bias_scales: &[u8],
    bias_global_scale: f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let (bias0, bias1) = nvfp4_values2(bias_bytes, bias_scales, bias_global_scale, col0 as usize);
    let pre00 = affine_from_stored_product(acc[0], scale0, bias0);
    let pre01 = affine_from_stored_product(acc[1], scale0, bias1);
    let pre10 = affine_from_stored_product(acc[2], scale1, bias0);
    let pre11 = affine_from_stored_product(acc[3], scale1, bias1);
    let relu00 = max_f32(pre00, 0.0);
    let relu01 = max_f32(pre01, 0.0);
    let relu10 = max_f32(pre10, 0.0);
    let relu11 = max_f32(pre11, 0.0);
    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;

    store_f32x2_global(pre_activation, index0, pre00, pre01);
    store_f32x2_global(pre_activation, index1, pre10, pre11);
    store_f16x2_global(pre_activation_f16, index0, pre00, pre01);
    store_f16x2_global(pre_activation_f16, index1, pre10, pre11);
    store_f32x2_global(out, index0, crate::activation::centered_relu2_from_positive(relu00), crate::activation::centered_relu2_from_positive(relu01));
    store_f32x2_global(out, index1, crate::activation::centered_relu2_from_positive(relu10), crate::activation::centered_relu2_from_positive(relu11));
}

#[inline(always)]
fn store_acc_relu2_compact_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    pre_activation_f16: *mut u16,
    out: &mut DisjointSlice<f32>,
    bias_tile: &BiasSmem,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;

    let bias_col = tile.mma_col_offset(n_repeat) + tile.thread_in_group * 2;
    let (bias0, bias1) = load_f32x2_shared(bias_tile.as_ptr(), bias_col as usize);
    let pre00 = affine_from_stored_product(acc[0], scale0, bias0);
    let pre01 = affine_from_stored_product(acc[1], scale0, bias1);
    let pre10 = affine_from_stored_product(acc[2], scale1, bias0);
    let pre11 = affine_from_stored_product(acc[3], scale1, bias1);
    let relu00 = max_f32(pre00, 0.0);
    let relu01 = max_f32(pre01, 0.0);
    let relu10 = max_f32(pre10, 0.0);
    let relu11 = max_f32(pre11, 0.0);
    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;

    if !pre_activation_f16.is_null() {
        store_f16x2_global(pre_activation_f16, index0, pre00, pre01);
        store_f16x2_global(pre_activation_f16, index1, pre10, pre11);
    }
    store_f32x2_global(out, index0, crate::activation::centered_relu2_from_positive(relu00), crate::activation::centered_relu2_from_positive(relu01));
    store_f32x2_global(out, index1, crate::activation::centered_relu2_from_positive(relu10), crate::activation::centered_relu2_from_positive(relu11));
}

macro_rules! store_accumulator_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $out:expr) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        $out:expr
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out,
            );
        )+
        store_accumulator_rows!([$($rest),*], $tile, $params, $output_scale, $out);
    }};
}

macro_rules! store_accumulator_plain_rows {
    ($shape:tt, $tile:expr, $params:expr, $output_scale:expr, ($out:expr)) => {{
        store_accumulator_rows!($shape, $tile, $params, $output_scale, $out);
    }};
}

macro_rules! store_accumulator_exact_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, ($out:expr)) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($out:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_scaled_exact(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out,
            );
        )+
        store_accumulator_exact_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out)
        );
    }};
}

macro_rules! store_accumulator_gradient_rows_impl {
    ([], $tile:expr, $params:expr, $output_scale:expr, ($out:expr, $state:expr, $sums:expr)) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($out:expr, $state:expr, $sums:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_scaled_gradient(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out, $state, $sums,
            );
        )+
        store_accumulator_gradient_rows_impl!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out, $state, $sums)
        );
    }};
}

macro_rules! store_accumulator_csf_gradient_rows {
    ($shape:tt, $tile:expr, $params:expr, $scale:expr, ($out:expr, $gradient:expr, $sums:expr)) => {{
        let gradient_scalars = GradientScalars::new($gradient);
        store_accumulator_gradient_rows_impl!(
            $shape,
            $tile,
            $params,
            $scale,
            ($out, gradient_scalars, $sums)
        );
    }};
}

macro_rules! store_accumulator_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($out:expr, $local_amax:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_scaled_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out,
                $local_amax,
            );
        )+
        store_accumulator_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out, $local_amax)
        );
    }};
}

macro_rules! store_accumulator_relu2_backward_f16_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($pre_activation:expr, $out:expr, $local_amax:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_relu2_backward_f16_scaled_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $pre_activation,
                $out,
                $local_amax,
            );
        )+
        store_accumulator_relu2_backward_f16_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($pre_activation, $out, $local_amax)
        );
    }};
}

macro_rules! store_accumulator_symmetric_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($out:expr, $local_amax:expr, $mirror:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_scaled_symmetric_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out,
                $local_amax,
                $mirror,
            );
        )+
        store_accumulator_symmetric_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out, $local_amax, $mirror)
        );
    }};
}

macro_rules! store_accumulator_gram_factor_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($inputs:expr, $out:expr, $local_amax:expr, $mirror:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_gram_factor_symmetric_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $inputs,
                $out,
                $local_amax,
                $mirror,
            );
        )+
        store_accumulator_gram_factor_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($inputs, $out, $local_amax, $mirror)
        );
    }};
}

macro_rules! store_accumulator_polar_step_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($inputs:expr, $out:expr, $local_amax:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_polar_step_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $inputs,
                $out,
                $local_amax,
            );
        )+
        store_accumulator_polar_step_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($inputs, $out, $local_amax)
        );
    }};
}

macro_rules! store_accumulator_polar_step_row_sumsq_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($inputs:expr, $out:expr, $local_row_sums:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_polar_step_row_sumsq(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $inputs,
                $out,
                $local_row_sums,
            );
        )+
        store_accumulator_polar_step_row_sumsq_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($inputs, $out, $local_row_sums)
        );
    }};
}

macro_rules! store_accumulator_affine_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($out:expr, $bias_tile:expr, $output_stride:expr, $output_offset:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_affine_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $out,
                $bias_tile,
                $output_stride,
                $output_offset,
            );
        )+
        store_accumulator_affine_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out, $bias_tile, $output_stride, $output_offset)
        );
    }};
}

macro_rules! store_accumulator_residual_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($residual:expr, $bias_tile:expr, $add_residual:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_residual_scaled::<{ $add_residual }>(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $residual,
                $bias_tile,
            );
        )+
        store_accumulator_residual_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($residual, $bias_tile, $add_residual)
        );
    }};
}

macro_rules! store_accumulator_relu2_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($pre_activation:expr, $out:expr, $bias_bytes:expr, $bias_scales:expr, $bias_global_scale:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_relu2_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $pre_activation,
                $out,
                $bias_bytes,
                $bias_scales,
                $bias_global_scale,
            );
        )+
        store_accumulator_relu2_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($pre_activation, $out, $bias_bytes, $bias_scales, $bias_global_scale)
        );
    }};
}

macro_rules! store_accumulator_relu2_f16_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($pre_activation:expr, $pre_activation_f16:expr, $out:expr, $bias_bytes:expr, $bias_scales:expr, $bias_global_scale:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_relu2_f16_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $pre_activation,
                $pre_activation_f16,
                $out,
                $bias_bytes,
                $bias_scales,
                $bias_global_scale,
            );
        )+
        store_accumulator_relu2_f16_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($pre_activation, $pre_activation_f16, $out, $bias_bytes, $bias_scales, $bias_global_scale)
        );
    }};
}

macro_rules! store_accumulator_relu2_compact_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($pre_activation_f16:expr, $out:expr, $bias_tile:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_relu2_compact_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $pre_activation_f16,
                $out,
                $bias_tile,
            );
        )+
        store_accumulator_relu2_compact_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($pre_activation_f16, $out, $bias_tile)
        );
    }};
}

macro_rules! accumulate_scalar_b_row {
    ([], $a:expr, $scale_a:expr, []) => {{}};
    (
        [($n_repeat:tt, $acc:ident) $(, $acc_rest:tt)*],
        $a:expr,
        $scale_a:expr,
        [($b_repeat:tt, $b:expr, $scale_b:expr) $(, $b_rest:tt)* $(,)?]
    ) => {{
        mma_m16n8k64_mxf4nvf4_scale4x_ue4m3($a, $b, &mut $acc, $scale_a, $scale_b);
        accumulate_scalar_b_row!([$($acc_rest),*], $a, $scale_a, [$($b_rest),*]);
    }};
}

macro_rules! accumulate_scalar_b_entries {
    (
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_packs:expr,
        $a_scales:expr,
        $tile:expr,
        $k_atom:expr,
        $b_entries:tt
    ) => {{
        $(
            let a = load_a_fragments_swizzled($a_packs, $tile, $k_atom, $m_repeat);
            let scale_a = load_a_scale4_packed($a_scales, $tile, $k_atom, $m_repeat);
            accumulate_scalar_b_row!([$(($n_repeat, $acc)),+], a, scale_a, $b_entries);
        )+
    }};
}

macro_rules! with_b_entries_swizzled {
    (
        $run:ident,
        $b_packs:expr,
        $b_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($n_axis:tt),+]
        $(, $arg:tt)*
    ) => {
        with_b_entries_swizzled_go!(
            $run,
            $b_packs,
            $b_scales,
            $tile,
            $k_atom,
            [],
            [$($n_axis),+]
            $(, $arg)*
        );
    };
}

macro_rules! with_b_entries_swizzled_go {
    (
        $run:ident,
        $b_packs:expr,
        $b_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($entries:tt)*],
        []
        $(, $arg:tt)*
    ) => {
        $run!([$($entries)*] $(, $arg)*);
    };
    (
        $run:ident,
        $b_packs:expr,
        $b_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($entries:tt)*],
        [$n_repeat:tt]
        $(, $arg:tt)*
    ) => {{
        let b = load_b_fragments_swizzled($b_packs, $tile, $k_atom, $n_repeat);
        let scale_b = load_b_scale4_packed($b_scales, $tile, $k_atom, $n_repeat);
        $run!([$($entries)* ($n_repeat, b, scale_b),] $(, $arg)*);
    }};
    (
        $run:ident,
        $b_packs:expr,
        $b_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($entries:tt)*],
        [$n0:tt, $n1:tt $(, $n_rest:tt)*]
        $(, $arg:tt)*
    ) => {{
        let (b0, b1) = if mma_n_atoms_are_adjacent($tile.n_layout, $tile.warp_n, $n0, $n1) {
            let pair = load_b_fragment_pair_ldmatrix($b_packs, $tile, $k_atom, $n0);
            (pair[0], pair[1])
        } else {
            (
                load_b_fragments_swizzled($b_packs, $tile, $k_atom, $n0),
                load_b_fragments_swizzled($b_packs, $tile, $k_atom, $n1),
            )
        };
        let scale_b0 = load_b_scale4_packed($b_scales, $tile, $k_atom, $n0);
        let scale_b1 = load_b_scale4_packed($b_scales, $tile, $k_atom, $n1);
        with_b_entries_swizzled_go!(
            $run,
            $b_packs,
            $b_scales,
            $tile,
            $k_atom,
            [
                $($entries)*
                ($n0, b0, scale_b0),
                ($n1, b1, scale_b1),
            ],
            [$($n_rest),*]
            $(, $arg)*
        );
    }};
}

macro_rules! accumulate_k_atom_scalar_b {
    (
        $shape:tt,
        $a_packs:expr,
        $b_packs:expr,
        $a_scales:expr,
        $b_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($n_axis:tt),+]
    ) => {{
        with_b_entries_swizzled!(
            accumulate_k_atom_scalar_b_with_entries,
            $b_packs,
            $b_scales,
            $tile,
            $k_atom,
            [$($n_axis),+],
            $shape,
            $a_packs,
            $a_scales,
            $tile,
            $k_atom
        );
    }};
}

macro_rules! accumulate_k_atom_scalar_b_with_entries {
    (
        $b_entries:tt,
        $shape:tt,
        $a_packs:expr,
        $a_scales:expr,
        $tile:expr,
        $k_atom:expr
    ) => {{
        accumulate_scalar_b_entries!($shape, $a_packs, $a_scales, $tile, $k_atom, $b_entries);
    }};
}

macro_rules! accumulate_staged_k_atoms {
    (
        $accumulate_k_atom:ident,
        $shape:tt,
        $a_packs:expr,
        $b_packs:expr,
        $a_scales:expr,
        $b_scales:expr,
        $tile:expr,
        $empty_bar:expr,
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$($k_atom:tt),+]
    ) => {{
        $($accumulate_k_atom!($k_atom);)+
        unsafe {
            fence_proxy_async_shared_cta();
            let _ = mbarrier_arrive($empty_bar as *const Barrier);
        }
    }};
}

const fn repeat_shape_supported(m_repeat: u32, n_repeat: u32) -> usize {
    let acc_count = m_repeat * n_repeat;
    if m_repeat > 0
        && n_repeat > 0
        && acc_count > 0
        && acc_count <= NVFP4_MAX_ACCUMULATORS_PER_THREAD
        && WARP_TILES_M > 0
        && WARP_TILES_N > 0
        && MMA_THREADS_PER_BLOCK <= 256
        && TMA_NVFP4_THREADS_PER_BLOCK <= 288
        && TMA_NVFP4_STATIC_SMEM_BYTES <= MAX_STATIC_SMEM_BYTES
    {
        1
    } else {
        0
    }
}

const _: [(); 1] = [(); repeat_shape_supported(M_REPEAT, N_REPEAT)];

macro_rules! dispatch_accumulator_shape {
    ($run:ident $(, $arg:expr)* $(,)?) => {{
        nvfp4_warp_issue_axes!(dispatch_accumulator_shape_for_axes, $run $(, $arg)*);
    }};
}

macro_rules! dispatch_accumulator_shape_for_axes {
    ([$($m_axis:tt),+], [$($n_axis:tt),+], $run:ident $(, $arg:expr)* $(,)?) => {
        build_accumulator_shape!($run, [$($m_axis),+], [$($n_axis),+] $(, $arg)*);
    };
}

macro_rules! build_accumulator_shape {
    (
        $run:ident,
        [$($m_repeat:tt),+],
        [$($n_repeat:tt),+]
        $(, $arg:expr)* $(,)?
    ) => {
        nvfp4_accumulator_pool!(
            build_accumulator_shape_with_pool,
            $run,
            [$($m_repeat),+],
            [$($n_repeat),+]
            $(, $arg)*
        );
    };
}

macro_rules! build_accumulator_shape_with_pool {
    (
        [$($acc_pool:ident),+],
        $run:ident,
        [$($m_repeat:tt),+],
        [$($n_repeat:tt),+]
        $(, $arg:expr)* $(,)?
    ) => {
        build_accumulator_rows!(
            $run,
            [$($m_repeat),+],
            [],
            [$($m_repeat),+],
            [$($n_repeat),+],
            [$($acc_pool),+]
            $(, $arg)*
        );
    };
}

macro_rules! build_accumulator_rows {
    (
        $run:ident,
        [$($m_all:tt),+],
        [$($rows:tt)*],
        [],
        [$($n_all:tt),+],
        [$($acc_pool:ident),*]
        $(, $arg:expr)* $(,)?
    ) => {
        $run!([$($m_all),+], [$($n_all),+], [$($rows)*] $(, $arg)*);
    };
    (
        $run:ident,
        [$($m_all:tt),+],
        [$($rows:tt)*],
        [$m_repeat:tt $(, $m_rest:tt)*],
        [$($n_all:tt),+],
        [$($acc_pool:ident),+]
        $(, $arg:expr)* $(,)?
    ) => {
        build_accumulator_cols!(
            $run,
            [$($m_all),+],
            [$($rows)*],
            $m_repeat,
            [],
            [$($n_all),+],
            [$($n_all),+],
            [$($m_rest),*],
            [$($acc_pool),+]
            $(, $arg)*
        );
    };
}

macro_rules! build_accumulator_cols {
    (
        $run:ident,
        [$($m_all:tt),+],
        [$($rows:tt)*],
        $m_repeat:tt,
        [$($cols:tt)*],
        [],
        [$($n_all:tt),+],
        [$($m_rest:tt),*],
        [$($acc_pool:ident),*]
        $(, $arg:expr)* $(,)?
    ) => {
        build_accumulator_rows!(
            $run,
            [$($m_all),+],
            [$($rows)* ($m_repeat, [$($cols)*]),],
            [$($m_rest),*],
            [$($n_all),+],
            [$($acc_pool),*]
            $(, $arg)*
        );
    };
    (
        $run:ident,
        [$($m_all:tt),+],
        [$($rows:tt)*],
        $m_repeat:tt,
        [$($cols:tt)*],
        [$n_repeat:tt $(, $n_rest:tt)*],
        [$($n_all:tt),+],
        [$($m_rest:tt),*],
        [$acc:ident $(, $acc_rest:ident)*]
        $(, $arg:expr)* $(,)?
    ) => {
        build_accumulator_cols!(
            $run,
            [$($m_all),+],
            [$($rows)*],
            $m_repeat,
            [$($cols)* ($n_repeat, $acc),],
            [$($n_rest),*],
            [$($n_all),+],
            [$($m_rest),*],
            [$($acc_rest),*]
            $(, $arg)*
        );
    };
}

macro_rules! scale_mode_for_store {
    (store_accumulator_csf_gradient_rows) => {
        3u32
    };
    (store_accumulator_csf_exact_rows) => {
        3u32
    };
    (store_accumulator_relu2_backward_f16_amax_rows) => {
        3u32
    };
    (store_accumulator_affine_rows) => {
        2u32
    };
    (store_accumulator_residual_rows) => {
        2u32
    };
    (store_accumulator_relu2_rows) => {
        2u32
    };
    (store_accumulator_relu2_f16_rows) => {
        2u32
    };
    (store_accumulator_relu2_compact_rows) => {
        2u32
    };
    ($other:ident) => {
        0u32
    };
}
macro_rules! store_accumulator_csf_exact_rows {($($args:tt)*)=>{store_accumulator_exact_rows!($($args)*);};}
macro_rules! inline_producer_for_store {
    (store_accumulator_residual_rows) => {
        true
    };
    ($other:ident) => {
        false
    };
}
macro_rules! run_tma_nvfp4_full_tile_shape_body {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $store:ident,
        $store_args:tt $(,)?
    ) => {{
        let a_tma = $a_tma;
        let b_tma = $b_tma;
        let a_scale_tma = $a_scale_tma;
        let b_scale_tma = $b_scale_tma;
        let tile = $tile;
        let params = $params;
        let a_packs_base = $a_packs_base;
        let b_packs_base = $b_packs_base;
        let a_scales_base = $a_scales_base;
        let b_scales_base = $b_scales_base;
        let tma_bars = $tma_bars;
        let empty_bars = $empty_bars;

        const SCALE_MODE:u32=scale_mode_for_store!($store);
        const INLINE_PRODUCER:bool=inline_producer_for_store!($store);
        const PIPE_STAGES:u32=if INLINE_PRODUCER{2}else{TMA_PIPELINE_STAGES};
        const A_STAGE:usize=if SCALE_MODE!=0{288}else{A_SCALES};
        const B_STAGE:usize=if SCALE_MODE!=0{288}else{B_SCALES};
        let k_tile_count = if routed_output_tile_active(tile, params) { routed_k_tile_count(tile, params) } else { 0 };
        let mut remaining_k=routed_k_mask(tile,params);
        macro_rules! issue_stage {
            ($kt:expr) => {{
                let kt = $kt;
                let stage = kt%PIPE_STAGES;
                let k_base=if params.route_mode==ROUTE_TOKEN_M_FEATURE_K
                    ||params.route_mode==ROUTE_FEATURE_M_TOKEN_K||params.route_mode==ROUTE_FEATURE_N_TOKEN_K{
                    let selected=remaining_k.take_first();
                    selected*TILE_K
                }else{kt*TILE_K};
                if SCALE_MODE != 0 {
                    stage_csf_tile(a_tma, b_tma, a_scale_tma, b_scale_tma, tile, k_base, params,
                        stage_ptr(a_packs_base,stage,A_PACKS),stage_ptr(b_packs_base,stage,B_PACKS),
                        stage_ptr(a_scales_base,stage,A_STAGE),stage_ptr(b_scales_base,stage,B_STAGE),stage_barrier(tma_bars,stage));
                } else {
                    stage_tiles_full_tma_nvfp4(a_tma, b_tma, a_scale_tma, b_scale_tma, tile, k_base, params,
                        stage_ptr(a_packs_base,stage,A_PACKS),stage_ptr(b_packs_base,stage,B_PACKS),
                        stage_ptr(a_scales_base,stage,A_STAGE),stage_ptr(b_scales_base,stage,B_STAGE),stage_barrier(tma_bars,stage));
                }
            }};
        }
        if INLINE_PRODUCER&&thread::threadIdx_x()==0{
            let mut kt=0;
            while kt<PIPE_STAGES&&kt<k_tile_count{issue_stage!(kt);kt+=1;}
        }
        if !INLINE_PRODUCER&&thread::threadIdx_x()>=MMA_THREADS_PER_BLOCK{
            if thread::threadIdx_x()==MMA_THREADS_PER_BLOCK{
                let mut kt=0;
                while kt<k_tile_count{
                    if kt>=PIPE_STAGES{
                        wait_mbarrier_parity(stage_barrier(empty_bars,kt%PIPE_STAGES),((kt/PIPE_STAGES)&1)^1);
                    }
                    issue_stage!(kt);
                    kt+=1;
                }
            }
        }else{
            $($(
                let mut $acc = [0.0f32; 4];
            )+)+

            macro_rules! accumulate_staged_tile_tma_scales {
                ($a_packs:expr, $b_packs:expr, $a_scales:expr, $b_scales:expr, $empty_bar:expr) => {{
                    macro_rules! accumulate_k_atom {
                        ($k_atom:expr) => {{
                            accumulate_k_atom_scalar_b!(
                                [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
                                $a_packs,
                                $b_packs,
                                $a_scales,
                                $b_scales,
                                tile,
                                $k_atom,
                                [$($n_axis),+]
                            );
                        }};
                    }

                    macro_rules! accumulate_k_axis {
                        ($k_axis:tt) => {{
                            accumulate_staged_k_atoms!(
                                accumulate_k_atom,
                                [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
                                $a_packs,
                                $b_packs,
                                $a_scales,
                                $b_scales,
                                tile,
                                $empty_bar,
                                [$($m_axis),+],
                                [$($n_axis),+],
                                $k_axis
                            );
                        }};
                    }

                    nvfp4_k_atom_axis!(accumulate_k_axis);
                }};
            }

            let mut k_tile = 0;
            while k_tile < k_tile_count {
                let stage = k_tile%PIPE_STAGES;
                let full_phase = (k_tile/PIPE_STAGES)&1;
                let tma_bar = stage_barrier(tma_bars, stage);
                let empty_bar = stage_barrier(empty_bars, stage);
                wait_mbarrier_parity(tma_bar, full_phase);
                let ascale=stage_ptr(a_scales_base,stage,A_STAGE);
                let bscale=stage_ptr(b_scales_base,stage,B_STAGE);
                if SCALE_MODE!=0{
                    const B_PAIRS:usize=N_REPEAT.div_ceil(2)as usize;
                    let first=tile.col_base%128+tile.mma_col_offset(0);let half=(first>>3)&1;
                    let b_raw=load_hardware_scales_raw::<B_PAIRS,1>(bscale,first);
                    let bs=decode_hardware_scales::<B_PAIRS,false>(b_raw,half);
                    let a_raw=load_hardware_scales_raw::<{M_REPEAT as usize},1>(ascale,tile.warp_m*M_PER_WARP);
                    let asf=decode_hardware_scales::<{M_REPEAT as usize},true>(a_raw,0);
                    accumulate_staged_tile_tma_scales!(stage_ptr(a_packs_base,stage,A_PACKS),stage_ptr(b_packs_base,stage,B_PACKS),asf,bs,empty_bar);
                }else{
                    let bs=load_native_scales::<{N_REPEAT.div_ceil(2) as usize},false>(bscale,tile);
                    let asf=load_native_scales::<{M_REPEAT as usize},true>(ascale,tile);
                    accumulate_staged_tile_tma_scales!(stage_ptr(a_packs_base,stage,A_PACKS),stage_ptr(b_packs_base,stage,B_PACKS),asf,bs,empty_bar);
                }
                let next=k_tile+PIPE_STAGES;
                if INLINE_PRODUCER&&thread::threadIdx_x()==0&&next<k_tile_count{
                    wait_mbarrier_parity(stage_barrier(empty_bars,next%PIPE_STAGES),((next/PIPE_STAGES)&1)^1);
                    issue_stage!(next);
                }
                k_tile += 1;
            }

            let output_scale = OutputScale::new(params);
            $store!(
                [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
                tile,
                params,
                output_scale,
                $store_args
            );
        }
    }};
}

macro_rules! run_tma_nvfp4_full_tile_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_plain_rows,
            ($out),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_exact_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_exact_rows,
            ($out),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_amax_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr,
        $local_amax:expr $(,)?)
    => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_amax_rows,
            ($out, $local_amax),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_symmetric_amax_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr,
        $local_amax:expr,
        $mirror:expr $(,)?)
    => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_symmetric_amax_rows,
            ($out, $local_amax, $mirror),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_gram_factor_amax_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $inputs:expr,
        $out:expr,
        $local_amax:expr,
        $mirror:expr $(,)?)
    => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_gram_factor_amax_rows,
            ($inputs, $out, $local_amax, $mirror),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_polar_step_amax_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $inputs:expr,
        $out:expr,
        $local_amax:expr $(,)?)
    => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_polar_step_amax_rows,
            ($inputs, $out, $local_amax),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_polar_step_row_sumsq_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $inputs:expr,
        $out:expr,
        $local_row_sums:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_polar_step_row_sumsq_rows,
            ($inputs, $out, $local_row_sums),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_affine_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr,
        $bias_tile:expr,
        $output_stride:expr,
        $output_offset:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_affine_rows,
            ($out, $bias_tile, $output_stride, $output_offset),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_residual_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $residual:expr,
        $bias_tile:expr,
        $add_residual:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_residual_rows,
            ($residual, $bias_tile, $add_residual),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $pre_activation:expr,
        $out:expr,
        $bias_bytes:expr,
        $bias_scales:expr,
        $bias_global_scale:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_relu2_rows,
            (
                $pre_activation,
                $out,
                $bias_bytes,
                $bias_scales,
                $bias_global_scale
            ),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_f16_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $pre_activation:expr,
        $pre_activation_f16:expr,
        $out:expr,
        $bias_bytes:expr,
        $bias_scales:expr,
        $bias_global_scale:expr $(,)?
    ) => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_relu2_f16_rows,
            (
                $pre_activation,
                $pre_activation_f16,
                $out,
                $bias_bytes,
                $bias_scales,
                $bias_global_scale
            ),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_compact_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $pre_activation_f16:expr,
        $out:expr,
        $bias_tile:expr $(,)?)
    => {{
        run_tma_nvfp4_full_tile_shape_body!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_relu2_compact_rows,
            (
                $pre_activation_f16,
                $out,
                $bias_tile
            ),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_exact {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_exact_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_symmetric_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_symmetric_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_gram_factor_amax {
    ($($arg:expr),+ $(,)?) => {
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_gram_factor_amax_shape, $($arg),+);
    };
}

macro_rules! run_tma_nvfp4_full_tile_polar_step_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_polar_step_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_polar_step_row_sumsq {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_polar_step_row_sumsq_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_affine {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_affine_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_residual {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_residual_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2 {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_relu2_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_f16 {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_relu2_f16_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_compact {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_relu2_compact_shape, $($arg),+);
    }};
}

macro_rules! run_csf_tile_shape_store {
    ($($args:tt)*) => {run_tma_nvfp4_full_tile_shape_body!($($args)*);};
}

macro_rules! run_csf_exact_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr $(,)?
    ) => {{
        run_csf_tile_shape_store!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_csf_exact_rows,
            ($out),
        );
    }};
}

macro_rules! run_csf_exact { ($($arg:expr),+ $(,)?) => {{ dispatch_accumulator_shape!(run_csf_exact_shape, $($arg),+); }}; }
macro_rules! run_csf_gradient_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $out:expr, $gradient:expr, $sums:expr $(,)?
    ) => {{
        run_csf_tile_shape_store!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_csf_gradient_rows,
            ($out, $gradient, $sums),
        );
    }};
}

macro_rules! run_csf_gradient { ($($arg:expr),+ $(,)?) => {{ dispatch_accumulator_shape!(run_csf_gradient_shape, $($arg),+); }}; }
macro_rules! run_csf_relu2_backward_f16_amax_shape {
    (
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$(($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?])),+ $(,)?],
        $a_tma:expr,
        $b_tma:expr,
        $a_scale_tma:expr,
        $b_scale_tma:expr,
        $tile:expr,
        $params:expr,
        $a_packs_base:expr,
        $b_packs_base:expr,
        $a_scales_base:expr,
        $b_scales_base:expr,
        $tma_bars:expr,
        $empty_bars:expr,
        $pre_activation:expr,
        $out:expr,
        $local_amax:expr $(,)?)
    => {{
        run_csf_tile_shape_store!(
            [$($m_axis),+],
            [$($n_axis),+],
            [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
            $a_tma,
            $b_tma,
            $a_scale_tma,
            $b_scale_tma,
            $tile,
            $params,
            $a_packs_base,
            $b_packs_base,
            $a_scales_base,
            $b_scales_base,
            $tma_bars,
            $empty_bars,
            store_accumulator_relu2_backward_f16_amax_rows,
            ($pre_activation, $out, $local_amax),
        );
    }};
}

macro_rules! run_csf_relu2_backward_f16_amax { ($($arg:expr),+ $(,)?) => {{ dispatch_accumulator_shape!(run_csf_relu2_backward_f16_amax_shape, $($arg),+); }}; }

