use core::marker::PhantomData;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TmaSwizzle {
    None,
    Swizzle64B,
    Swizzle128B,
}

pub trait U4SmemSwizzle {
    const TMA_SWIZZLE: TmaSwizzle;

    fn u32_index(logical_u32: u32) -> u32;
}

pub trait U4SmemLayout {
    const PACKS_PER_ROW: u32;
    const TMA_SWIZZLE: TmaSwizzle;

    fn u32_index(row: u32, pack: u32) -> usize;
    fn ldmatrix_chunk_start_u32(row: u32, pack: u32) -> usize;
}

pub struct NoSwizzle;

impl U4SmemSwizzle for NoSwizzle {
    const TMA_SWIZZLE: TmaSwizzle = TmaSwizzle::None;

    #[inline(always)]
    fn u32_index(logical_u32: u32) -> u32 {
        logical_u32
    }
}

pub struct PositionIndependentSwizzle64B;

impl U4SmemSwizzle for PositionIndependentSwizzle64B {
    const TMA_SWIZZLE: TmaSwizzle = TmaSwizzle::Swizzle64B;

    #[inline(always)]
    fn u32_index(logical_u32: u32) -> u32 {
        // CUTE's as_position_independent_swizzle_tensor recasts the TMA
        // Swizzle<2,4,3> from bytes to uint4_t, yielding a u32-pack swizzle
        // over bits [5:6] controlled by bits [8:9] of the logical U4 index.
        logical_u32 ^ ((logical_u32 & 0x60) >> 3)
    }
}

pub struct PositionIndependentSwizzle128B;

impl U4SmemSwizzle for PositionIndependentSwizzle128B {
    const TMA_SWIZZLE: TmaSwizzle = TmaSwizzle::Swizzle128B;

    #[inline(always)]
    fn u32_index(logical_u32: u32) -> u32 {
        logical_u32 ^ ((logical_u32 & 0xe0) >> 3)
    }
}

pub struct Sm120KMajorSwizzle<const PACKS_PER_ROW: u32>;

impl<const PACKS_PER_ROW: u32> U4SmemSwizzle for Sm120KMajorSwizzle<PACKS_PER_ROW> {
    const TMA_SWIZZLE: TmaSwizzle = if PACKS_PER_ROW % 32 == 0 {
        TmaSwizzle::Swizzle128B
    } else {
        TmaSwizzle::Swizzle64B
    };

    #[inline(always)]
    fn u32_index(logical_u32: u32) -> u32 {
        if PACKS_PER_ROW % 32 == 0 {
            PositionIndependentSwizzle128B::u32_index(logical_u32)
        } else {
            PositionIndependentSwizzle64B::u32_index(logical_u32)
        }
    }
}

pub struct KMajorU4<const PACKS_PER_ROW: u32, S: U4SmemSwizzle>(PhantomData<S>);

impl<const PACKS_PER_ROW: u32, S: U4SmemSwizzle> KMajorU4<PACKS_PER_ROW, S> {
    #[inline(always)]
    pub fn u32_index(row: u32, pack: u32) -> usize {
        S::u32_index(row * PACKS_PER_ROW + pack) as usize
    }

    #[inline(always)]
    pub fn ldmatrix_chunk_start_u32(row: u32, pack: u32) -> usize {
        S::u32_index(row * PACKS_PER_ROW + (pack & !3)) as usize
    }
}

impl<const PACKS_PER_ROW: u32, S: U4SmemSwizzle> U4SmemLayout for KMajorU4<PACKS_PER_ROW, S> {
    const PACKS_PER_ROW: u32 = PACKS_PER_ROW;
    const TMA_SWIZZLE: TmaSwizzle = S::TMA_SWIZZLE;

    #[inline(always)]
    fn u32_index(row: u32, pack: u32) -> usize {
        Self::u32_index(row, pack)
    }

    #[inline(always)]
    fn ldmatrix_chunk_start_u32(row: u32, pack: u32) -> usize {
        Self::ldmatrix_chunk_start_u32(row, pack)
    }
}

pub trait ShapeExpr {
    type Coord;

    const EXTENT: u32;

    fn flatten(coord: Self::Coord) -> u32;
    fn unflatten(index: u32) -> Self::Coord;
    fn contains(coord: Self::Coord) -> bool;
}

pub struct Axis<const EXTENT: u32>;

impl<const EXTENT: u32> Axis<EXTENT> {
    pub const EXTENT: u32 = EXTENT;

    #[inline(always)]
    pub const fn flatten(coord: u32) -> u32 {
        coord
    }

    #[inline(always)]
    pub const fn unflatten(index: u32) -> u32 {
        index
    }

    #[inline(always)]
    pub const fn contains(coord: u32) -> bool {
        coord < EXTENT
    }
}

impl<const EXTENT: u32> ShapeExpr for Axis<EXTENT> {
    type Coord = u32;

    const EXTENT: u32 = EXTENT;

    #[inline(always)]
    fn flatten(coord: Self::Coord) -> u32 {
        Self::flatten(coord)
    }

    #[inline(always)]
    fn unflatten(index: u32) -> Self::Coord {
        Self::unflatten(index)
    }

    #[inline(always)]
    fn contains(coord: Self::Coord) -> bool {
        Self::contains(coord)
    }
}

pub struct ShapePair<Left, Right>(PhantomData<(Left, Right)>);

impl<Left: ShapeExpr, Right: ShapeExpr> ShapePair<Left, Right> {
    pub const EXTENT: u32 = Left::EXTENT * Right::EXTENT;

    #[inline(always)]
    pub fn flatten(coord: (Left::Coord, Right::Coord)) -> u32 {
        let (left, right) = coord;
        Left::flatten(left) * Right::EXTENT + Right::flatten(right)
    }

    #[inline(always)]
    pub fn unflatten(index: u32) -> (Left::Coord, Right::Coord) {
        (
            Left::unflatten(index / Right::EXTENT),
            Right::unflatten(index % Right::EXTENT),
        )
    }

    #[inline(always)]
    pub fn contains(coord: (Left::Coord, Right::Coord)) -> bool {
        let (left, right) = coord;
        Left::contains(left) && Right::contains(right)
    }
}

impl<Left: ShapeExpr, Right: ShapeExpr> ShapeExpr for ShapePair<Left, Right> {
    type Coord = (Left::Coord, Right::Coord);

    const EXTENT: u32 = Self::EXTENT;

    #[inline(always)]
    fn flatten(coord: Self::Coord) -> u32 {
        Self::flatten(coord)
    }

    #[inline(always)]
    fn unflatten(index: u32) -> Self::Coord {
        Self::unflatten(index)
    }

    #[inline(always)]
    fn contains(coord: Self::Coord) -> bool {
        Self::contains(coord)
    }
}

pub struct Recompose<Source, Target>(PhantomData<(Source, Target)>);

impl<Source: ShapeExpr, Target: ShapeExpr> Recompose<Source, Target> {
    pub const SAME_EXTENT: bool = Source::EXTENT == Target::EXTENT;

    #[inline(always)]
    pub fn map(coord: Source::Coord) -> Target::Coord {
        Target::unflatten(Source::flatten(coord))
    }

    #[inline(always)]
    pub fn inverse(coord: Target::Coord) -> Source::Coord {
        Source::unflatten(Target::flatten(coord))
    }
}

#[macro_export]
macro_rules! nvfp4_cute_shape {
    (($head:tt, $($tail:tt),+)) => {
        $crate::nvfp4::cute::ShapePair<
            $crate::nvfp4_cute_shape!($head),
            $crate::nvfp4_cute_shape!(($($tail),+))
        >
    };
    (($single:tt)) => {
        $crate::nvfp4_cute_shape!($single)
    };
    ($extent:expr) => {
        $crate::nvfp4::cute::Axis<{ $extent }>
    };
}

pub trait MmaAtomShape {
    const M: u32;
    const N: u32;
    const K: u32;
}

pub struct Sm120Nvfp4MmaAtom;

impl Sm120Nvfp4MmaAtom {
    pub const M: u32 = 16;
    pub const N: u32 = 8;
    pub const K: u32 = 64;
}

impl MmaAtomShape for Sm120Nvfp4MmaAtom {
    const M: u32 = Self::M;
    const N: u32 = Self::N;
    const K: u32 = Self::K;
}

pub const N_LAYOUT_PAIR_INTERLEAVED: u32 = 0;
pub const N_LAYOUT_WARP_CONTIGUOUS: u32 = 1;
pub const N_LAYOUT_REPEAT_INTERLEAVED: u32 = 2;
pub const N_LAYOUT_PAIR_REVERSE_GROUPS: u32 = 3;
pub const N_LAYOUT_PAIR_OUTER_GROUPS: u32 = 4;

pub struct Sm120Nvfp4WarpMma<const M_REPEAT: u32, const N_REPEAT: u32, const N_LAYOUT: u32>(
    PhantomData<()>,
);

impl<const M_REPEAT: u32, const N_REPEAT: u32, const N_LAYOUT: u32>
    Sm120Nvfp4WarpMma<M_REPEAT, N_REPEAT, N_LAYOUT>
{
    pub const M_PER_WARP: u32 = M_REPEAT * Sm120Nvfp4MmaAtom::M;
    pub const N_PER_WARP: u32 = N_REPEAT * Sm120Nvfp4MmaAtom::N;
    pub const ACC_COUNT: usize = (M_REPEAT * N_REPEAT) as usize;

    #[inline(always)]
    pub const fn n_atom<const WARP_TILES_N: u32>(warp_n: u32, n_repeat: u32) -> u32 {
        match N_LAYOUT {
            N_LAYOUT_PAIR_INTERLEAVED => {
                pair_grouped_n_atom(WARP_TILES_N, N_REPEAT, warp_n, n_repeat, GroupOrder::Normal)
            }
            N_LAYOUT_WARP_CONTIGUOUS => warp_n * N_REPEAT + n_repeat,
            N_LAYOUT_REPEAT_INTERLEAVED => n_repeat * WARP_TILES_N + warp_n,
            N_LAYOUT_PAIR_REVERSE_GROUPS => pair_grouped_n_atom(
                WARP_TILES_N,
                N_REPEAT,
                warp_n,
                n_repeat,
                GroupOrder::Reverse,
            ),
            N_LAYOUT_PAIR_OUTER_GROUPS => {
                pair_grouped_n_atom(WARP_TILES_N, N_REPEAT, warp_n, n_repeat, GroupOrder::Outer)
            }
            _ => 0,
        }
    }

    #[inline(always)]
    pub const fn n_atoms_are_adjacent<const WARP_TILES_N: u32>(
        warp_n: u32,
        n0: u32,
        n1: u32,
    ) -> bool {
        Self::n_atom::<WARP_TILES_N>(warp_n, n1) == Self::n_atom::<WARP_TILES_N>(warp_n, n0) + 1
    }
}

#[derive(Clone, Copy)]
enum GroupOrder {
    Normal,
    Reverse,
    Outer,
}

const fn pair_grouped_n_atom(
    warp_tiles_n: u32,
    n_repeat_count: u32,
    warp_n: u32,
    n_repeat: u32,
    group_order: GroupOrder,
) -> u32 {
    if n_repeat_count < 2 {
        return warp_n * n_repeat_count + n_repeat;
    }

    let source_group = n_repeat / 2;
    let lane = n_repeat - source_group * 2;
    let groups = ceil_div_u32(n_repeat_count, 2);
    let ordered_group = group_order_at(groups, source_group, group_order);
    ((ordered_group * warp_tiles_n + warp_n) * 2) + lane
}

const fn group_order_at(groups: u32, position: u32, order: GroupOrder) -> u32 {
    match order {
        GroupOrder::Normal => position,
        GroupOrder::Reverse => groups - 1 - position,
        GroupOrder::Outer => outer_at(groups, position),
    }
}

const fn outer_at(extent: u32, position: u32) -> u32 {
    if position & 1 == 0 {
        position / 2
    } else {
        extent - 1 - position / 2
    }
}

const fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        ((value - 1) / divisor) + 1
    }
}

pub struct Sm120ScaleLayout;

impl Sm120ScaleLayout {
    pub const VECTOR_SIZE: u32 = 16;
    pub const MN_BLOCK: u32 = 128;
    pub const MN_FAST: u32 = 32;
    pub const GROUPS_PER_K_ATOM: u32 = 4;
    pub const K_ATOM: u32 = Self::VECTOR_SIZE * Self::GROUPS_PER_K_ATOM;
    pub const TMA_WIDTH_U16: u32 = 8;
    pub const PAD_BYTE: u8 = 0x38;
    pub const BYTES_PER_MN_K_ATOM: u32 = Self::MN_BLOCK * Self::GROUPS_PER_K_ATOM;
    pub const TMA_ROW_BYTES: u32 = Self::TMA_WIDTH_U16 * 2;
    pub const TMA_SWIZZLE_ROW_BYTES: u32 = 64;
    pub const TMA_SWIZZLE_ROW_FACTOR: u32 = Self::TMA_SWIZZLE_ROW_BYTES / Self::TMA_ROW_BYTES;

    #[inline(always)]
    pub const fn k_groups(k_dim: u32) -> u32 {
        k_dim / Self::VECTOR_SIZE
    }

    #[inline(always)]
    pub const fn k_atoms(k_dim: u32) -> u32 {
        Self::k_groups(k_dim) / Self::GROUPS_PER_K_ATOM
    }

    #[inline(always)]
    pub const fn padded_mn_extent(mn_extent: u32) -> u32 {
        mn_extent.div_ceil(Self::MN_BLOCK) * Self::MN_BLOCK
    }

    #[inline(always)]
    pub const fn packed_len(mn_extent: u32, k_dim: u32) -> usize {
        ((mn_extent / Self::MN_BLOCK) * Self::k_atoms(k_dim) * Self::BYTES_PER_MN_K_ATOM) as usize
    }

    #[inline(always)]
    pub const fn tma_height(mn_extent: u32, k_dim: u32) -> u32 {
        Self::packed_len(mn_extent, k_dim) as u32 / Self::TMA_ROW_BYTES
    }

    #[inline(always)]
    pub const fn tma_y(mn_base: u32, k_base: u32, mn_extent: u32) -> i32 {
        let mn_blocks = mn_extent / Self::MN_BLOCK;
        let mn_block = mn_base / Self::MN_BLOCK;
        let k_atom = k_base / Self::K_ATOM;
        ((k_atom * mn_blocks + mn_block) * Self::MN_FAST) as i32
    }

    #[inline(always)]
    pub const fn tma_y_padded(mn_base: u32, k_base: u32, mn_extent: u32) -> i32 {
        let padded_mn_extent = Self::padded_mn_extent(mn_extent);
        let aligned_mn_base = (mn_base / Self::MN_BLOCK) * Self::MN_BLOCK;
        Self::tma_y(aligned_mn_base, k_base, padded_mn_extent)
    }

    #[inline(always)]
    pub const fn block_major_tma_y(mn_base: u32, k_base: u32, _mn_extent: u32, k_dim: u32) -> i32 {
        let k_atoms = Self::k_atoms(k_dim);
        let mn_block = mn_base / Self::MN_BLOCK;
        let k_atom = k_base / Self::K_ATOM;
        ((mn_block * k_atoms + k_atom) * Self::MN_FAST) as i32
    }

    #[inline(always)]
    pub const fn block_major_tma_y_padded(
        mn_base: u32,
        k_base: u32,
        mn_extent: u32,
        k_dim: u32,
    ) -> i32 {
        let padded_mn_extent = Self::padded_mn_extent(mn_extent);
        let aligned_mn_base = (mn_base / Self::MN_BLOCK) * Self::MN_BLOCK;
        Self::block_major_tma_y(aligned_mn_base, k_base, padded_mn_extent, k_dim)
    }

    #[inline(always)]
    pub const fn byte_offset(mn: u32, k_group: u32, mn_extent: u32) -> usize {
        let mn_blocks = mn_extent / Self::MN_BLOCK;
        let mn_block = mn / Self::MN_BLOCK;
        let mn_in_block = mn - mn_block * Self::MN_BLOCK;
        let mn_fast = mn_in_block % Self::MN_FAST;
        let mn_slow = mn_in_block / Self::MN_FAST;
        let k_atom = k_group / Self::GROUPS_PER_K_ATOM;
        let k_in_atom = k_group % Self::GROUPS_PER_K_ATOM;

        (k_atom * mn_blocks * Self::BYTES_PER_MN_K_ATOM
            + mn_block * Self::BYTES_PER_MN_K_ATOM
            + mn_slow * Self::MN_FAST * Self::GROUPS_PER_K_ATOM
            + mn_fast * Self::GROUPS_PER_K_ATOM
            + k_in_atom) as usize
    }

    #[inline(always)]
    pub const fn block_major_byte_offset(
        mn: u32,
        k_group: u32,
        _mn_extent: u32,
        k_dim: u32,
    ) -> usize {
        let k_atoms = Self::k_atoms(k_dim);
        let k_atom = k_group / Self::GROUPS_PER_K_ATOM;
        let k_in_atom = k_group % Self::GROUPS_PER_K_ATOM;

        Self::block_major_u32_word_offset(mn, k_atom, k_atoms) * core::mem::size_of::<u32>()
            + k_in_atom as usize
    }

    #[inline(always)]
    pub const fn block_major_u32_word_offset(mn: u32, k_atom: u32, k_atoms: u32) -> usize {
        let mn_block = mn / Self::MN_BLOCK;
        let mn_in_block = mn - mn_block * Self::MN_BLOCK;
        let mn_fast = mn_in_block % Self::MN_FAST;
        let mn_slow = mn_in_block / Self::MN_FAST;

        ((mn_block * k_atoms + k_atom) * Self::MN_BLOCK + mn_slow * Self::MN_FAST + mn_fast)
            as usize
    }

    #[inline(always)]
    pub const fn u32_word_offset(mn: u32, k_atom: u32, mn_extent: u32) -> usize {
        Self::byte_offset(mn, k_atom * Self::GROUPS_PER_K_ATOM, mn_extent)
            / core::mem::size_of::<u32>()
    }
}
