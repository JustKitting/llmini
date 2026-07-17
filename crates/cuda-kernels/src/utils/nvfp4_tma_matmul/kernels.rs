use cuda_device::{
    DisjointSlice, SharedArray,
    barrier::{
        Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_expect_tx,
        mbarrier_init, mbarrier_try_wait_parity,
    },
    convert::cvt_f16x2_f32,
    cuda_module, kernel, launch_bounds, ptx_asm, thread,
    tma::TmaDescriptor,
};

use crate::f16_tc_matmul::convert::{load_f16x2_global, load_f32x2_global_read_only};
use crate::float_ptx::{abs_f32, fma_f32, max_f32, sqrt_f32};
use crate::nvfp4::nvfp4_values2;
use crate::{
    block_reduce::block_sum_shared_f32,
    warp_reduce::{eighth_warp_sum_f32, thread_lane_warp, warp_max_nonnegative_f32},
};

use super::cute::{
    KMajorU4, N_LAYOUT_WARP_CONTIGUOUS, Sm120KMajorSwizzle, Sm120Nvfp4MmaAtom, Sm120Nvfp4WarpMma,
    Sm120ScaleLayout,
};
use super::load::E4M3_ONE_PACKED4;

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
}

const MMA_M: u32 = Sm120Nvfp4MmaAtom::M;
const MMA_N: u32 = Sm120Nvfp4MmaAtom::N;
const MMA_K: u32 = Sm120Nvfp4MmaAtom::K;
const SCALE_MN_BLOCK: u32 = Sm120ScaleLayout::MN_BLOCK;

include!(concat!(env!("OUT_DIR"), "/nvfp4_config.rs"));

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
const LINEAR3_ROW_SUMS_PER_THREAD: usize = (M_REPEAT * 2) as usize;
const LINEAR3_ROW_SUMS_PER_CTA: usize = (TILE_M * WARP_TILES_N) as usize;
const TMA_PRODUCER_THREADS: u32 = 32;
const TMA_PRODUCER_THREAD: u32 = MMA_THREADS_PER_BLOCK;
const TMA_PIPELINE_STAGES: u32 = NVFP4_TMA_STAGES;
const TMA_PIPELINE_STAGES_USIZE: usize = TMA_PIPELINE_STAGES as usize;
const MAX_STATIC_SMEM_BYTES: u32 = 96 * 1024;
pub const TMA_NVFP4_THREADS_PER_BLOCK: u32 = MMA_THREADS_PER_BLOCK + TMA_PRODUCER_THREADS;

#[inline]
pub const fn tma_nvfp4_output_amax_chunks(token_count: u32, output_dim: u32) -> u32 {
    ceil_div_u32(token_count, TILE_M) * ceil_div_u32(output_dim, TILE_N) * WARPS_PER_BLOCK
}

#[inline]
pub const fn tma_nvfp4_symmetric_output_amax_chunks(dim: u32) -> u32 {
    let tiles = ceil_div_u32(dim, TILE_M);
    (tiles * (tiles + 1) / 2) * WARPS_PER_BLOCK
}

const TMA_NVFP4_MAINLOOP_BYTES: u32 =
    A_TILE_BYTES + B_TILE_BYTES + A_SCALE_TILE_BYTES + B_SCALE_TILE_BYTES;

const _: [(); TILE_K as usize] = [(); (K_ATOMS * MMA_K) as usize];
const _: [(); TILE_M as usize] = [(); (WARP_TILES_M * M_PER_WARP) as usize];
const _: [(); TILE_N as usize] = [(); (WARP_TILES_N * N_PER_WARP) as usize];
const _: [(); MMA_THREADS_PER_BLOCK as usize] = [(); (WARPS_PER_BLOCK * 32) as usize];
const _: [(); TMA_NVFP4_THREADS_PER_BLOCK as usize] =
    [(); (WARPS_PER_BLOCK * 32 + TMA_PRODUCER_THREADS) as usize];
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
const A_SCALE_TILE_BYTES: u32 = (A_SCALES * core::mem::size_of::<u32>()) as u32;
const B_SCALE_TILE_BYTES: u32 = (B_SCALES * core::mem::size_of::<u32>()) as u32;
const TMA_NVFP4_STATIC_SMEM_BYTES: u32 = TMA_PIPELINE_STAGES * TMA_NVFP4_MAINLOOP_BYTES;
const A_PACKS_STAGED: usize = A_PACKS * TMA_PIPELINE_STAGES_USIZE;
const B_PACKS_STAGED: usize = B_PACKS * TMA_PIPELINE_STAGES_USIZE;
const A_SCALES_STAGED: usize = A_SCALES * TMA_PIPELINE_STAGES_USIZE;
const B_SCALES_STAGED: usize = B_SCALES * TMA_PIPELINE_STAGES_USIZE;

type APacksSmemStages = SharedArray<u32, A_PACKS_STAGED, 1024>;
type BPacksSmemStages = SharedArray<u32, B_PACKS_STAGED, 1024>;
type AScalesSmemStages = SharedArray<u32, A_SCALES_STAGED, 128>;
type BScalesSmemStages = SharedArray<u32, B_SCALES_STAGED, 128>;
type BarrierSmemStages = SharedArray<Barrier, TMA_PIPELINE_STAGES_USIZE, 8>;
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
fn wait_mbarrier_parity(bar: *mut Barrier, parity: u32) {
    while unsafe { !mbarrier_try_wait_parity(bar as *const Barrier, parity) } {}
}

#[inline(always)]
const fn pipeline_stage(k_tile: u32) -> u32 {
    k_tile % TMA_PIPELINE_STAGES
}

#[inline(always)]
const fn pipeline_phase(k_tile: u32) -> u32 {
    (k_tile / TMA_PIPELINE_STAGES) & 1
}

#[inline(always)]
const fn producer_empty_phase(k_tile: u32) -> u32 {
    pipeline_phase(k_tile) ^ 1
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
    scale_a: u32,
    scale_b: u32,
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
             %11, {0, 0}, %12, {0, 0};
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
            in("r") scale_a,
            in("r") scale_b,
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
        ptx_asm!(
            "st.global.v2.f32 [%0], {%1, %2};",
            in("l") ptr as u64,
            in("f") x,
            in("f") y,
        );
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
            let scalar_b = unsafe { *(params.b_global_scale as usize as *const f32) };
            let scalar_a = if params.global_scale_mode == 1 {
                unsafe { *(params.a_global_scale as usize as *const f32) }
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
                unsafe { *((self.a_global_scale as usize as *const f32).add(row as usize)) };
            self.weight * row_scale * self.scalar_b
        } else {
            self.weight * self.scalar_a * self.scalar_b
        }
    }
}

#[inline(always)]
fn scale_tma_y(mn_base: u32, k_base: u32, mn_extent: u32, k_dim: u32) -> i32 {
    Sm120ScaleLayout::block_major_tma_y_padded(mn_base, k_base, mn_extent, k_dim)
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
    if thread::threadIdx_x() == TMA_PRODUCER_THREAD {
        unsafe {
            mbarrier_arrive_expect_tx(tma_bar as *const Barrier, 1, TMA_NVFP4_MAINLOOP_BYTES);
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
            options(register_only),
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
    Sm120ScaleLayout::u32_word_offset(mn, k_atom, mn_extent)
}

#[inline(always)]
fn load_a_scale4_packed(a_scales: *const u32, tile: CtaTile, k_atom: u32, m_repeat: u32) -> u32 {
    if tile.thread_in_group >= 2 {
        return E4M3_ONE_PACKED4;
    }
    let row = tile.warp_m * M_PER_WARP
        + m_repeat * MMA_M
        + tile.group
        + if (tile.thread_in_group & 1) == 1 {
            MMA_M / 2
        } else {
            0
        };
    let scale_row = (tile.row_base % SCALE_MN_BLOCK) + row;
    unsafe { *a_scales.add(scale_packed_u32_index(scale_row, k_atom, A_SCALE_MN)) }
}

#[inline(always)]
fn load_b_scale4_packed(b_scales: *const u32, tile: CtaTile, k_atom: u32, n_repeat: u32) -> u32 {
    if tile.thread_in_group != 0 {
        return E4M3_ONE_PACKED4;
    }
    let col = tile.mma_col_offset(n_repeat) + tile.group;
    let scale_col = (tile.col_base % SCALE_MN_BLOCK) + col;
    unsafe { *b_scales.add(scale_packed_u32_index(scale_col, k_atom, B_SCALE_MN)) }
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
    if col0 + 1 >= output_dim {
        return;
    }

    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;
    let (pre00, pre01) = load_f16x2_global(pre_activation.as_ptr(), index0 as usize);
    let (pre10, pre11) = load_f16x2_global(pre_activation.as_ptr(), index1 as usize);
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

#[inline(always)]
fn store_acc_linear3_scaled_amax(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    source: &[f32],
    action: &[f32],
    out: &mut DisjointSlice<f32>,
    bound_scale: f32,
    a_scale: f32,
    b_scale: f32,
    c_scale: f32,
    local_amax: &mut f32,
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;
    let (source00, source01) = load_f32x2_global_read_only(source.as_ptr(), index0 as usize);
    let (source10, source11) = load_f32x2_global_read_only(source.as_ptr(), index1 as usize);
    let (action00, action01) = load_f32x2_global_read_only(action.as_ptr(), index0 as usize);
    let (action10, action11) = load_f32x2_global_read_only(action.as_ptr(), index1 as usize);
    let current00 = acc[0] * scale0;
    let current01 = acc[1] * scale0;
    let current10 = acc[2] * scale1;
    let current11 = acc[3] * scale1;
    let value00 = fma_f32(
        a_scale,
        source00 * bound_scale,
        fma_f32(b_scale, action00, c_scale * current00),
    );
    let value01 = fma_f32(
        a_scale,
        source01 * bound_scale,
        fma_f32(b_scale, action01, c_scale * current01),
    );
    let value10 = fma_f32(
        a_scale,
        source10 * bound_scale,
        fma_f32(b_scale, action10, c_scale * current10),
    );
    let value11 = fma_f32(
        a_scale,
        source11 * bound_scale,
        fma_f32(b_scale, action11, c_scale * current11),
    );
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

#[inline(always)]
fn store_acc_linear3_scaled_row_sumsq(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    source: &[f32],
    action: &[f32],
    out: &mut DisjointSlice<f32>,
    bound_scale: f32,
    a_scale: f32,
    b_scale: f32,
    c_scale: f32,
    local_row_sums: &mut [f32; LINEAR3_ROW_SUMS_PER_THREAD],
) {
    let row0 = tile.mma_row_base(m_repeat) + tile.group;
    let row1 = row0 + 8;
    let col0 = tile.mma_col_base(n_repeat) + tile.thread_in_group * 2;
    let output_dim = params.output_dim;
    if col0 + 1 >= output_dim {
        return;
    }

    let index0 = row0 * output_dim + col0;
    let index1 = row1 * output_dim + col0;
    let (source00, source01) = load_f32x2_global_read_only(source.as_ptr(), index0 as usize);
    let (source10, source11) = load_f32x2_global_read_only(source.as_ptr(), index1 as usize);
    let (action00, action01) = load_f32x2_global_read_only(action.as_ptr(), index0 as usize);
    let (action10, action11) = load_f32x2_global_read_only(action.as_ptr(), index1 as usize);
    let current00 = acc[0] * scale0;
    let current01 = acc[1] * scale0;
    let current10 = acc[2] * scale1;
    let current11 = acc[3] * scale1;
    let value00 = fma_f32(
        a_scale,
        source00 * bound_scale,
        fma_f32(b_scale, action00, c_scale * current00),
    );
    let value01 = fma_f32(
        a_scale,
        source01 * bound_scale,
        fma_f32(b_scale, action01, c_scale * current01),
    );
    let value10 = fma_f32(
        a_scale,
        source10 * bound_scale,
        fma_f32(b_scale, action10, c_scale * current10),
    );
    let value11 = fma_f32(
        a_scale,
        source11 * bound_scale,
        fma_f32(b_scale, action11, c_scale * current11),
    );
    let sum0 = &mut local_row_sums[(m_repeat * 2) as usize];
    *sum0 = fma_f32(value00, value00, *sum0);
    *sum0 = fma_f32(value01, value01, *sum0);
    let sum1 = &mut local_row_sums[(m_repeat * 2 + 1) as usize];
    *sum1 = fma_f32(value10, value10, *sum1);
    *sum1 = fma_f32(value11, value11, *sum1);
    store_f32x2_global(out, index0, value00, value01);
    store_f32x2_global(out, index1, value10, value11);
}

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
    store_f32x2_global(
        out,
        row0 * output_dim + col0,
        affine_from_stored_product(acc[0], scale0, bias0),
        affine_from_stored_product(acc[1], scale0, bias1),
    );
    store_f32x2_global(
        out,
        row1 * output_dim + col0,
        affine_from_stored_product(acc[2], scale1, bias0),
        affine_from_stored_product(acc[3], scale1, bias1),
    );
}

#[inline(always)]
fn store_acc_residual_scaled(
    acc: [f32; 4],
    tile: CtaTile,
    m_repeat: u32,
    n_repeat: u32,
    params: Nvfp4GemmParams,
    scale0: f32,
    scale1: f32,
    residual: &mut DisjointSlice<f32>,
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

    let index00 = row0 * output_dim + col0;
    let index10 = row1 * output_dim + col0;
    let (bias0, bias1) = nvfp4_values2(bias_bytes, bias_scales, bias_global_scale, col0 as usize);
    let value00 = affine_from_stored_product(acc[0], scale0, bias0);
    let value01 = affine_from_stored_product(acc[1], scale0, bias1);
    let value10 = affine_from_stored_product(acc[2], scale1, bias0);
    let value11 = affine_from_stored_product(acc[3], scale1, bias1);
    let residual00 = unsafe { *residual.get_unchecked_mut(index00 as usize) };
    let residual01 = unsafe { *residual.get_unchecked_mut((index00 + 1) as usize) };
    let residual10 = unsafe { *residual.get_unchecked_mut(index10 as usize) };
    let residual11 = unsafe { *residual.get_unchecked_mut((index10 + 1) as usize) };
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
        relu00 * relu00,
        relu01 * relu01,
    );
    store_f32x2_global(
        out,
        row1 * output_dim + col0,
        relu10 * relu10,
        relu11 * relu11,
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
    store_f32x2_global(out, index0, relu00 * relu00, relu01 * relu01);
    store_f32x2_global(out, index1, relu10 * relu10, relu11 * relu11);
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

    if !pre_activation_f16.is_null() {
        store_f16x2_global(pre_activation_f16, index0, pre00, pre01);
        store_f16x2_global(pre_activation_f16, index1, pre10, pre11);
    }
    store_f32x2_global(out, index0, relu00 * relu00, relu01 * relu01);
    store_f32x2_global(out, index1, relu10 * relu10, relu11 * relu11);
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

macro_rules! store_accumulator_linear3_amax_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($source:expr, $action:expr, $out:expr, $bound_scale:expr, $a_scale:expr, $b_scale:expr, $c_scale:expr, $local_amax:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_linear3_scaled_amax(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $source,
                $action,
                $out,
                $bound_scale,
                $a_scale,
                $b_scale,
                $c_scale,
                $local_amax,
            );
        )+
        store_accumulator_linear3_amax_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($source, $action, $out, $bound_scale, $a_scale, $b_scale, $c_scale, $local_amax)
        );
    }};
}

macro_rules! store_accumulator_linear3_row_sumsq_rows {
    ([], $tile:expr, $params:expr, $output_scale:expr, $args:tt) => {};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $rest:tt)* $(,)?],
        $tile:expr,
        $params:expr,
        $output_scale:expr,
        ($source:expr, $action:expr, $out:expr, $bound_scale:expr, $a_scale:expr, $b_scale:expr, $c_scale:expr, $local_row_sums:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_linear3_scaled_row_sumsq(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $source,
                $action,
                $out,
                $bound_scale,
                $a_scale,
                $b_scale,
                $c_scale,
                $local_row_sums,
            );
        )+
        store_accumulator_linear3_row_sumsq_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($source, $action, $out, $bound_scale, $a_scale, $b_scale, $c_scale, $local_row_sums)
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
        ($out:expr, $bias_bytes:expr, $bias_scales:expr, $bias_global_scale:expr)
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
                $bias_bytes,
                $bias_scales,
                $bias_global_scale,
            );
        )+
        store_accumulator_affine_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($out, $bias_bytes, $bias_scales, $bias_global_scale)
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
        ($residual:expr, $bias_bytes:expr, $bias_scales:expr, $bias_global_scale:expr)
    ) => {{
        let row0 = $tile.mma_row_base($m_repeat) + $tile.group;
        let scale0 = $output_scale.row(row0);
        let scale1 = $output_scale.row(row0 + 8);
        $(
            store_acc_residual_scaled(
                $acc,
                $tile,
                $m_repeat,
                $n_repeat,
                $params,
                scale0,
                scale1,
                $residual,
                $bias_bytes,
                $bias_scales,
                $bias_global_scale,
            );
        )+
        store_accumulator_residual_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($residual, $bias_bytes, $bias_scales, $bias_global_scale)
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
        ($pre_activation_f16:expr, $out:expr, $bias_bytes:expr, $bias_scales:expr, $bias_global_scale:expr)
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
                $bias_bytes,
                $bias_scales,
                $bias_global_scale,
            );
        )+
        store_accumulator_relu2_compact_rows!(
            [$($rest),*],
            $tile,
            $params,
            $output_scale,
            ($pre_activation_f16, $out, $bias_bytes, $bias_scales, $bias_global_scale)
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

macro_rules! accumulate_preloaded_a_rows {
    ([], [], $b_entries:tt) => {{}};
    (
        [($m_repeat:tt, [$(($n_repeat:tt, $acc:ident)),+ $(,)?]) $(, $shape_rest:tt)* $(,)?],
        [($a_repeat:tt, $a:expr, $scale_a:expr) $(, $a_rest:tt)* $(,)?],
        $b_entries:tt
    ) => {{
        accumulate_scalar_b_row!([$(($n_repeat, $acc)),+], $a, $scale_a, $b_entries);
        accumulate_preloaded_a_rows!([$($shape_rest),*], [$($a_rest),*], $b_entries);
    }};
}

macro_rules! accumulate_preloaded_a_scalar_b_entries {
    ($shape:tt, $a_entries:tt, $b_entries:tt) => {{
        accumulate_preloaded_a_rows!($shape, $a_entries, $b_entries);
    }};
}

macro_rules! with_a_entries_swizzled {
    (
        $run:ident,
        $a_packs:expr,
        $a_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($m_axis:tt),+]
        $(, $arg:tt)*
    ) => {
        with_a_entries_swizzled_go!(
            $run,
            $a_packs,
            $a_scales,
            $tile,
            $k_atom,
            [],
            [$($m_axis),+]
            $(, $arg)*
        );
    };
}

macro_rules! with_a_entries_swizzled_go {
    (
        $run:ident,
        $a_packs:expr,
        $a_scales:expr,
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
        $a_packs:expr,
        $a_scales:expr,
        $tile:expr,
        $k_atom:expr,
        [$($entries:tt)*],
        [$m_repeat:tt $(, $m_rest:tt)*]
        $(, $arg:tt)*
    ) => {{
        let a = load_a_fragments_swizzled($a_packs, $tile, $k_atom, $m_repeat);
        let scale_a = load_a_scale4_packed($a_scales, $tile, $k_atom, $m_repeat);
        with_a_entries_swizzled_go!(
            $run,
            $a_packs,
            $a_scales,
            $tile,
            $k_atom,
            [$($entries)* ($m_repeat, a, scale_a),],
            [$($m_rest),*]
            $(, $arg)*
        );
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
        [$single_k:tt]
    ) => {{
        $accumulate_k_atom!($single_k);
        unsafe {
            fence_proxy_async_shared_cta();
            let _ = mbarrier_arrive($empty_bar as *const Barrier);
        }
    }};
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
        [$first_k:tt, $next_k:tt $(, $rest_k:tt)*]
    ) => {{
        macro_rules! preload_next_a_entries {
            ($a_entries:tt) => {{
                macro_rules! preload_next_b_entries {
                    ($b_entries:tt) => {{
                        $accumulate_k_atom!($first_k);
                        accumulate_staged_k_atoms_loaded!(
                            $shape,
                            $a_packs,
                            $b_packs,
                            $a_scales,
                            $b_scales,
                            $tile,
                            $empty_bar,
                            [$($m_axis),+],
                            [$($n_axis),+],
                            [$($rest_k),*],
                            $a_entries,
                            $b_entries
                        );
                    }};
                }

                with_b_entries_swizzled!(
                    preload_next_b_entries,
                    $b_packs,
                    $b_scales,
                    $tile,
                    $next_k,
                    [$($n_axis),+]
                );
            }};
        }

        with_a_entries_swizzled!(
            preload_next_a_entries,
            $a_packs,
            $a_scales,
            $tile,
            $next_k,
            [$($m_axis),+]
        );
    }};
}

macro_rules! accumulate_staged_k_atoms_loaded {
    (
        $shape:tt,
        $a_packs:expr,
        $b_packs:expr,
        $a_scales:expr,
        $b_scales:expr,
        $tile:expr,
        $empty_bar:expr,
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [],
        $a_entries:tt,
        $b_entries:tt
    ) => {{
        unsafe {
            fence_proxy_async_shared_cta();
            let _ = mbarrier_arrive($empty_bar as *const Barrier);
        }
        accumulate_preloaded_a_scalar_b_entries!($shape, $a_entries, $b_entries);
    }};
    (
        $shape:tt,
        $a_packs:expr,
        $b_packs:expr,
        $a_scales:expr,
        $b_scales:expr,
        $tile:expr,
        $empty_bar:expr,
        [$($m_axis:tt),+],
        [$($n_axis:tt),+],
        [$next_k:tt $(, $rest_k:tt)*],
        $a_entries:tt,
        $b_entries:tt
    ) => {{
        macro_rules! preload_next_a_entries {
            ($next_a_entries:tt) => {{
                macro_rules! preload_next_b_entries {
                    ($next_b_entries:tt) => {{
                        accumulate_preloaded_a_scalar_b_entries!(
                            $shape,
                            $a_entries,
                            $b_entries
                        );
                        accumulate_staged_k_atoms_loaded!(
                            $shape,
                            $a_packs,
                            $b_packs,
                            $a_scales,
                            $b_scales,
                            $tile,
                            $empty_bar,
                            [$($m_axis),+],
                            [$($n_axis),+],
                            [$($rest_k),*],
                            $next_a_entries,
                            $next_b_entries
                        );
                    }};
                }

                with_b_entries_swizzled!(
                    preload_next_b_entries,
                    $b_packs,
                    $b_scales,
                    $tile,
                    $next_k,
                    [$($n_axis),+]
                );
            }};
        }

        with_a_entries_swizzled!(
            preload_next_a_entries,
            $a_packs,
            $a_scales,
            $tile,
            $next_k,
            [$($m_axis),+]
        );
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

        if thread::threadIdx_x() >= MMA_THREADS_PER_BLOCK {
            let mut k_base = 0;
            let mut k_tile = 0;
            while k_base < params.input_dim {
                let stage = pipeline_stage(k_tile);
                let empty_phase = producer_empty_phase(k_tile);
                let tma_bar = stage_barrier(tma_bars, stage);
                let empty_bar = stage_barrier(empty_bars, stage);
                wait_mbarrier_parity(empty_bar, empty_phase);
                stage_tiles_full_tma_nvfp4(
                    a_tma,
                    b_tma,
                    a_scale_tma,
                    b_scale_tma,
                    tile,
                    k_base,
                    params,
                    stage_ptr(a_packs_base, stage, A_PACKS),
                    stage_ptr(b_packs_base, stage, B_PACKS),
                    stage_ptr(a_scales_base, stage, A_SCALES),
                    stage_ptr(b_scales_base, stage, B_SCALES),
                    tma_bar,
                );
                k_base += TILE_K;
                k_tile += 1;
            }

            let mut tail_tile = k_tile;
            let mut tail_count = 0;
            while tail_count < TMA_PIPELINE_STAGES {
                let stage = pipeline_stage(tail_tile);
                let empty_phase = producer_empty_phase(tail_tile);
                wait_mbarrier_parity(stage_barrier(empty_bars, stage), empty_phase);
                tail_tile += 1;
                tail_count += 1;
            }
        } else {
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

            let mut k_base = 0;
            let mut k_tile = 0;
            while k_base < params.input_dim {
                let stage = pipeline_stage(k_tile);
                let full_phase = pipeline_phase(k_tile);
                let tma_bar = stage_barrier(tma_bars, stage);
                let empty_bar = stage_barrier(empty_bars, stage);
                wait_mbarrier_parity(tma_bar, full_phase);
                accumulate_staged_tile_tma_scales!(
                    stage_ptr(a_packs_base, stage, A_PACKS),
                    stage_ptr(b_packs_base, stage, B_PACKS),
                    stage_ptr(a_scales_base, stage, A_SCALES),
                    stage_ptr(b_scales_base, stage, B_SCALES),
                    empty_bar
                );
                k_base += TILE_K;
                k_tile += 1;
            }

            $store!(
                [$(($m_repeat, [$(($n_repeat, $acc)),+])),+],
                tile,
                params,
                OutputScale::new(params),
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

macro_rules! run_tma_nvfp4_full_tile_relu2_backward_f16_amax_shape {
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
            store_accumulator_relu2_backward_f16_amax_rows,
            ($pre_activation, $out, $local_amax),
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

macro_rules! run_tma_nvfp4_full_tile_linear3_amax_shape {
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
        $source:expr,
        $action:expr,
        $out:expr,
        $bound_scale:expr,
        $a_scale:expr,
        $b_scale:expr,
        $c_scale:expr,
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
            store_accumulator_linear3_amax_rows,
            ($source, $action, $out, $bound_scale, $a_scale, $b_scale, $c_scale, $local_amax),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_linear3_row_sumsq_shape {
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
        $source:expr,
        $action:expr,
        $out:expr,
        $bound_scale:expr,
        $a_scale:expr,
        $b_scale:expr,
        $c_scale:expr,
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
            store_accumulator_linear3_row_sumsq_rows,
            ($source, $action, $out, $bound_scale, $a_scale, $b_scale, $c_scale, $local_row_sums),
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
            store_accumulator_affine_rows,
            ($out, $bias_bytes, $bias_scales, $bias_global_scale),
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
            store_accumulator_residual_rows,
            ($residual, $bias_bytes, $bias_scales, $bias_global_scale),
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
        $bias_bytes:expr,
        $bias_scales:expr,
        $bias_global_scale:expr $(,)?)
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
                $bias_bytes,
                $bias_scales,
                $bias_global_scale
            ),
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_relu2_backward_f16_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(
            run_tma_nvfp4_full_tile_relu2_backward_f16_amax_shape,
            $($arg),+
        );
    }};
}

macro_rules! run_tma_nvfp4_full_tile_symmetric_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_symmetric_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_linear3_amax {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_linear3_amax_shape, $($arg),+);
    }};
}

macro_rules! run_tma_nvfp4_full_tile_linear3_row_sumsq {
    ($($arg:expr),+ $(,)?) => {{
        dispatch_accumulator_shape!(run_tma_nvfp4_full_tile_linear3_row_sumsq_shape, $($arg),+);
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
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_kernel(
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
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new(thread_id);
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
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        mut output_chunk_amax: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_relu2_backward_f16_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        pre_activation: &[u16],
        mut out: DisjointSlice<f32>,
        mut output_chunk_amax: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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
        run_tma_nvfp4_full_tile_relu2_backward_f16_amax!(
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

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_symmetric_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        mut out: DisjointSlice<f32>,
        mut output_chunk_amax: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
    ) {
        let cta = thread::blockIdx_x();
        let tiles = params.output_dim / TILE_N;
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
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new_at(thread_id, NVFP4_N_LAYOUT, tile_row, tile_col);
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

        if thread_id < MMA_THREADS_PER_BLOCK {
            let warp_amax = warp_max_nonnegative_f32(local_amax);
            if thread_id & 31 == 0 {
                let chunk = cta * WARPS_PER_BLOCK + thread_id / 32;
                unsafe {
                    *output_chunk_amax.get_unchecked_mut(chunk as usize) = warp_amax;
                }
            }
        }
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_linear3_amax_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        source: &[f32],
        action: &[f32],
        mut out: DisjointSlice<f32>,
        bound_amax: &[f32],
        mut output_chunk_amax: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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
        let bound_scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let mut local_amax = 0.0;
        run_tma_nvfp4_full_tile_linear3_amax!(
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
            source,
            action,
            &mut out,
            bound_scale,
            a_scale,
            b_scale,
            c_scale,
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

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
    #[allow(unused_variables)]
    pub fn nvfp4_gemm_tma_linear3_row_sumsq_kernel(
        a_tma: *const TmaDescriptor,
        b_tma: *const TmaDescriptor,
        a_scale_tma: *const TmaDescriptor,
        b_scale_tma: *const TmaDescriptor,
        source: &[f32],
        action: &[f32],
        mut out: DisjointSlice<f32>,
        bound_amax: &[f32],
        mut row_sumsq_partials: DisjointSlice<f32>,
        params: Nvfp4GemmParams,
        a_scale: f32,
        b_scale: f32,
        c_scale: f32,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;
        static mut ROW_SUMS: SharedArray<f32, LINEAR3_ROW_SUMS_PER_CTA> = SharedArray::UNINIT;

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
        let bound_scale = if bound > 1.0 {
            1.0 / sqrt_f32(bound)
        } else {
            1.0
        };
        let mut local_row_sums = [0.0_f32; LINEAR3_ROW_SUMS_PER_THREAD];
        run_tma_nvfp4_full_tile_linear3_row_sumsq!(
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
            source,
            action,
            &mut out,
            bound_scale,
            a_scale,
            b_scale,
            c_scale,
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

        if thread_id < TILE_M {
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

    #[kernel]
    pub fn nvfp4_gemm_tma_row_sumsq_reduce_kernel(
        row_sumsq_partials: &[f32],
        mut row_sumsq: DisjointSlice<f32>,
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
            unsafe {
                *row_sumsq.get_unchecked_mut(row as usize) = sum;
            }
        }
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
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
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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

        run_tma_nvfp4_full_tile_affine!(
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
            bias_bytes,
            bias_scales,
            bias_global_scale[0],
        );
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
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
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

        let tile = CtaTile::new_with_n_layout(thread_id, N_LAYOUT_WARP_CONTIGUOUS);
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

        run_tma_nvfp4_full_tile_residual!(
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
            &mut residual,
            bias_bytes,
            bias_scales,
            bias_global_scale[0],
        );
    }

    #[kernel]
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
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
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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
    #[cfg_attr(nvfp4_launch_bounds_64, launch_bounds(64, 1))]
    #[cfg_attr(nvfp4_launch_bounds_96, launch_bounds(96, 1))]
    #[cfg_attr(nvfp4_launch_bounds_128, launch_bounds(128, 1))]
    #[cfg_attr(nvfp4_launch_bounds_160, launch_bounds(160, 1))]
    #[cfg_attr(nvfp4_launch_bounds_192, launch_bounds(192, 1))]
    #[cfg_attr(nvfp4_launch_bounds_224, launch_bounds(224, 1))]
    #[cfg_attr(nvfp4_launch_bounds_256, launch_bounds(256, 1))]
    #[cfg_attr(nvfp4_launch_bounds_288, launch_bounds(288, 1))]
    #[cfg_attr(nvfp4_launch_bounds_384, launch_bounds(384, 1))]
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
        params: Nvfp4GemmParams,
    ) {
        let thread_id = thread::threadIdx_x();

        static mut TMA_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut EMPTY_BARS: BarrierSmemStages = SharedArray::UNINIT;
        static mut A_SCALES_SM: AScalesSmemStages = SharedArray::UNINIT;
        static mut B_SCALES_SM: BScalesSmemStages = SharedArray::UNINIT;
        static mut A_PACKS_SM: APacksSmemStages = SharedArray::UNINIT;
        static mut B_PACKS_SM: BPacksSmemStages = SharedArray::UNINIT;

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
            bias_bytes,
            bias_scales,
            bias_global_scale[0],
        );
    }
}
