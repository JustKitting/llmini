use crate::mma::projection::Nvfp4ProjectionParams;
use crate::mma::projection::load_bytes::{
    E4M3_ONE_PACKED4, load_packed8, load_packed8_aligned, load_scale4, load_scale4_aligned,
};
use crate::nvfp4::csf::scale4 as csf_scale4;

use super::super::tile::{
    NVFP4_PROJECTION_CTA_M, NVFP4_PROJECTION_CTA_N, NVFP4_PROJECTION_CTA_PACKS_PER_ROW,
    Nvfp4ProjectionCtaTile,
};

const MMA_K: u32 = 64;

macro_rules! pack_loader_pair {
    ($checked:ident, $aligned:ident, $base:ident, |$row:ident, $col:ident, $params:ident| $bounds:expr, $index:expr) => {
        #[inline(always)]
        pub(super) fn $checked(
            bytes: &[u8],
            tile: Nvfp4ProjectionCtaTile,
            offset: u32,
            k_base: u32,
            $params: &Nvfp4ProjectionParams,
        ) -> u32 {
            let ($row, $col) = pack_coords(offset, tile.$base, k_base);
            if $bounds {
                load_packed8(bytes, $index as usize)
            } else {
                0
            }
        }

        #[inline(always)]
        pub(super) fn $aligned(
            bytes: &[u8],
            tile: Nvfp4ProjectionCtaTile,
            offset: u32,
            k_base: u32,
            $params: &Nvfp4ProjectionParams,
        ) -> u32 {
            let ($row, $col) = pack_coords(offset, tile.$base, k_base);
            load_packed8_aligned(bytes, $index as usize)
        }
    };
}

macro_rules! scale_loader_pair {
    ($checked:ident, $aligned:ident, $base:ident, $rows_per_atom:expr, $load:ident, $aligned_load:ident, |$row:ident, $col:ident, $params:ident| $bounds:expr, $index:expr) => {
        #[inline(always)]
        pub(super) fn $checked(
            scales: &[u8],
            tile: Nvfp4ProjectionCtaTile,
            offset: u32,
            k_base: u32,
            $params: &Nvfp4ProjectionParams,
        ) -> u32 {
            let ($row, $col) = scale_coords(offset, tile.$base, $rows_per_atom, k_base);
            if $bounds {
                $load(scales, $index as usize)
            } else {
                E4M3_ONE_PACKED4
            }
        }

        #[inline(always)]
        pub(super) fn $aligned(
            scales: &[u8],
            tile: Nvfp4ProjectionCtaTile,
            offset: u32,
            k_base: u32,
            $params: &Nvfp4ProjectionParams,
        ) -> u32 {
            let ($row, $col) = scale_coords(offset, tile.$base, $rows_per_atom, k_base);
            $aligned_load(scales, $index as usize)
        }
    };
}

pack_loader_pair!(
    load_a_pack,
    load_a_pack_aligned,
    row_base,
    |global_row, global_col, params| global_row < params.token_count
        && global_col + 7 < params.input_dim,
    global_row * params.input_dim + global_col
);
pack_loader_pair!(
    load_b_pack,
    load_b_pack_aligned,
    col_base,
    |global_col, global_k, params| global_col < params.output_dim
        && global_k + 7 < params.input_dim,
    global_col * params.input_dim + global_k
);
scale_loader_pair!(
    load_a_scale,
    load_a_scale_aligned,
    row_base,
    NVFP4_PROJECTION_CTA_M,
    load_scale4,
    load_scale4_aligned,
    |global_row, scale_k_base, params| global_row < params.token_count
        && scale_k_base < params.input_dim,
    (global_row * params.input_dim + scale_k_base) / 16
);
scale_loader_pair!(
    load_b_scale,
    load_b_scale_aligned,
    col_base,
    NVFP4_PROJECTION_CTA_N,
    csf_scale4,
    csf_scale4,
    |global_col, scale_k_base, params| global_col < params.output_dim
        && scale_k_base < params.input_dim,
    global_col * (params.input_dim / 16) + scale_k_base / 16
);

#[inline(always)]
fn pack_coords(offset: u32, row_base: u32, k_base: u32) -> (u32, u32) {
    let row = offset / NVFP4_PROJECTION_CTA_PACKS_PER_ROW;
    let pack = offset - row * NVFP4_PROJECTION_CTA_PACKS_PER_ROW;
    (row_base + row, k_base + pack * 8)
}

#[inline(always)]
fn scale_coords(offset: u32, row_base: u32, rows_per_atom: u32, k_base: u32) -> (u32, u32) {
    let k_atom = offset / rows_per_atom;
    let row = offset - k_atom * rows_per_atom;
    (row_base + row, k_base + k_atom * MMA_K)
}
