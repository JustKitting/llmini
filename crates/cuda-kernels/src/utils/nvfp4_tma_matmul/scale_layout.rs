use super::cute::Sm120ScaleLayout;

pub const SM120_SCALE_VECTOR_SIZE: usize = Sm120ScaleLayout::VECTOR_SIZE as usize;
pub const SM120_SCALE_MN_BLOCK: usize = Sm120ScaleLayout::MN_BLOCK as usize;
pub const SM120_SCALE_MN_FAST: usize = Sm120ScaleLayout::MN_FAST as usize;
pub const SM120_SCALE_GROUPS_PER_K_ATOM: usize = Sm120ScaleLayout::GROUPS_PER_K_ATOM as usize;
pub const SM120_SCALE_TMA_WIDTH_U16: usize = Sm120ScaleLayout::TMA_WIDTH_U16 as usize;
pub const SM120_SCALE_PAD_BYTE: u8 = Sm120ScaleLayout::PAD_BYTE;

const SM120_SCALE_TMA_ROW_BYTES: usize = Sm120ScaleLayout::TMA_ROW_BYTES as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sm120ScaleTmaShape {
    pub width_u16: u64,
    pub height: u64,
    pub row_stride_bytes: u64,
    pub tile_width_u16: u32,
    pub tile_height: u32,
}

pub fn sm120_scale_k_groups(k_dim: usize) -> usize {
    assert!(
        k_dim % SM120_SCALE_VECTOR_SIZE == 0,
        "NVFP4 scale K dimension must be divisible by {SM120_SCALE_VECTOR_SIZE}"
    );
    Sm120ScaleLayout::k_groups(k_dim as u32) as usize
}

pub const fn sm120_scale_padded_mn_extent(mn_extent: usize) -> usize {
    Sm120ScaleLayout::padded_mn_extent(mn_extent as u32) as usize
}

pub fn sm120_scale_tma_height(mn_extent: usize, k_dim: usize) -> usize {
    Sm120ScaleLayout::tma_height(mn_extent as u32, k_dim as u32) as usize
}

pub fn sm120_scale_tma_shape(
    mn_extent: usize,
    k_dim: usize,
    tile_mn: usize,
    tile_k: usize,
) -> Sm120ScaleTmaShape {
    validate_scale_shape(mn_extent, k_dim);
    validate_scale_shape(tile_mn, tile_k);
    Sm120ScaleTmaShape {
        width_u16: SM120_SCALE_TMA_WIDTH_U16 as u64,
        height: sm120_scale_tma_height(mn_extent, k_dim) as u64,
        row_stride_bytes: SM120_SCALE_TMA_ROW_BYTES as u64,
        tile_width_u16: SM120_SCALE_TMA_WIDTH_U16 as u32,
        tile_height: sm120_scale_tma_height(tile_mn, tile_k) as u32,
    }
}

pub fn sm120_scale_tma_shape_padded(
    mn_extent: usize,
    k_dim: usize,
    tile_mn: usize,
    tile_k: usize,
) -> Sm120ScaleTmaShape {
    validate_scale_shape(sm120_scale_padded_mn_extent(mn_extent), k_dim);
    validate_scale_shape(sm120_scale_padded_mn_extent(tile_mn), tile_k);
    Sm120ScaleTmaShape {
        width_u16: SM120_SCALE_TMA_WIDTH_U16 as u64,
        height: sm120_scale_tma_height(sm120_scale_padded_mn_extent(mn_extent), k_dim) as u64,
        row_stride_bytes: SM120_SCALE_TMA_ROW_BYTES as u64,
        tile_width_u16: SM120_SCALE_TMA_WIDTH_U16 as u32,
        tile_height: sm120_scale_tma_height(sm120_scale_padded_mn_extent(tile_mn), tile_k) as u32,
    }
}

pub fn sm120_scale_tma_y(mn_base: usize, k_base: usize, mn_extent: usize, k_dim: usize) -> i32 {
    assert!(
        mn_base % SM120_SCALE_MN_BLOCK == 0,
        "scale TMA MN base must be tile aligned"
    );
    assert!(mn_base < mn_extent, "scale TMA MN base is outside extent");
    assert!(
        k_base % (SM120_SCALE_VECTOR_SIZE * SM120_SCALE_GROUPS_PER_K_ATOM) == 0,
        "scale TMA K base must start at a 4-group scale atom"
    );
    assert!(
        mn_extent % SM120_SCALE_MN_BLOCK == 0,
        "SM120 scale layout requires MN extent divisible by {SM120_SCALE_MN_BLOCK}"
    );
    validate_scale_shape(mn_extent, k_dim);

    Sm120ScaleLayout::block_major_tma_y(
        mn_base as u32,
        k_base as u32,
        mn_extent as u32,
        k_dim as u32,
    )
}

pub fn sm120_scale_packed_len(mn_extent: usize, k_dim: usize) -> usize {
    validate_scale_shape(mn_extent, k_dim);
    Sm120ScaleLayout::packed_len(mn_extent as u32, k_dim as u32)
}

pub fn sm120_scale_byte_offset(mn: usize, k_group: usize, mn_extent: usize, k_dim: usize) -> usize {
    assert!(
        mn_extent % SM120_SCALE_MN_BLOCK == 0,
        "SM120 scale layout requires MN extent divisible by {SM120_SCALE_MN_BLOCK}"
    );
    assert!(mn < mn_extent, "MN index is outside the scale extent");
    validate_scale_shape(mn_extent, k_dim);

    Sm120ScaleLayout::block_major_byte_offset(
        mn as u32,
        k_group as u32,
        mn_extent as u32,
        k_dim as u32,
    )
}

pub fn sm120_scale_u32_word_offset(mn: usize, k_atom: usize, mn_extent: usize) -> usize {
    Sm120ScaleLayout::u32_word_offset(mn as u32, k_atom as u32, mn_extent as u32)
}

pub fn pack_sm120_scale_plane_compact(logical: &[u8], mn_extent: usize, k_dim: usize) -> Vec<u8> {
    pack_sm120_scale_plane(logical, mn_extent, k_dim, sm120_scale_k_groups(k_dim))
}

pub fn pack_sm120_scale_plane_compact_padded(
    logical: &[u8],
    mn_extent: usize,
    k_dim: usize,
) -> Vec<u8> {
    pack_sm120_scale_plane_padded(
        logical,
        mn_extent,
        k_dim,
        sm120_scale_k_groups(k_dim),
        SM120_SCALE_PAD_BYTE,
    )
}

pub fn pack_sm120_scale_plane(
    logical: &[u8],
    mn_extent: usize,
    k_dim: usize,
    logical_row_stride_bytes: usize,
) -> Vec<u8> {
    let (k_groups, _, _) = validate_scale_shape(mn_extent, k_dim);
    assert!(
        logical_row_stride_bytes >= k_groups,
        "logical scale row stride is smaller than the K scale group count"
    );
    if mn_extent > 0 {
        let required_len = (mn_extent - 1) * logical_row_stride_bytes + k_groups;
        assert!(
            logical.len() >= required_len,
            "logical scale input is shorter than the requested shape"
        );
    }

    let mut packed = vec![0; sm120_scale_packed_len(mn_extent, k_dim)];
    for mn in 0..mn_extent {
        let row_base = mn * logical_row_stride_bytes;
        for k_group in 0..k_groups {
            let dst = sm120_scale_byte_offset(mn, k_group, mn_extent, k_dim);
            packed[dst] = logical[row_base + k_group];
        }
    }
    packed
}

pub fn pack_sm120_scale_plane_padded(
    logical: &[u8],
    mn_extent: usize,
    k_dim: usize,
    logical_row_stride_bytes: usize,
    pad_byte: u8,
) -> Vec<u8> {
    let padded_mn_extent = sm120_scale_padded_mn_extent(mn_extent);
    let (k_groups, _, _) = validate_scale_shape(padded_mn_extent, k_dim);
    assert!(
        logical_row_stride_bytes >= k_groups,
        "logical scale row stride is smaller than the K scale group count"
    );
    if mn_extent > 0 {
        let required_len = (mn_extent - 1) * logical_row_stride_bytes + k_groups;
        assert!(
            logical.len() >= required_len,
            "logical scale input is shorter than the requested shape"
        );
    }

    let mut packed = vec![pad_byte; sm120_scale_packed_len(padded_mn_extent, k_dim)];
    for mn in 0..mn_extent {
        let row_base = mn * logical_row_stride_bytes;
        for k_group in 0..k_groups {
            let dst = sm120_scale_byte_offset(mn, k_group, padded_mn_extent, k_dim);
            packed[dst] = logical[row_base + k_group];
        }
    }
    packed
}

fn validate_scale_shape(mn_extent: usize, k_dim: usize) -> (usize, usize, usize) {
    assert!(
        mn_extent % SM120_SCALE_MN_BLOCK == 0,
        "SM120 scale layout requires MN extent divisible by {SM120_SCALE_MN_BLOCK}"
    );
    let k_groups = sm120_scale_k_groups(k_dim);
    assert!(
        k_groups % SM120_SCALE_GROUPS_PER_K_ATOM == 0,
        "SM120 scale layout requires K scale groups divisible by {SM120_SCALE_GROUPS_PER_K_ATOM}"
    );
    let k_atoms = k_groups / SM120_SCALE_GROUPS_PER_K_ATOM;
    let mn_blocks = mn_extent / SM120_SCALE_MN_BLOCK;
    (k_groups, k_atoms, mn_blocks)
}

/// Canonical compressed scale allocation, including exact raw-record capacity.
pub const fn sm120_csf_scale_storage_len(rows: usize, k: usize) -> usize {
    crate::nvfp4::csf::rowwise_encoded_len(rows, k)
}
