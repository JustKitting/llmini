use cuda_device::DisjointSlice;

use super::cta_tile::CtaMatmulDims;
use super::{CtaATile, CtaBTile};

macro_rules! causal_body {
    ($name:ident, $rhs:ty, $stage:path, $store:path) => {
        causal_body!(@typed $name, f32, $rhs, f32, $stage, $store);
    };
    (@typed $name:ident, $lhs:ty, $rhs:ty, $out:ty, $stage:path, $store:path) => {
        pub(super) fn $name(
            a: &[$lhs],
            b_t: &[$rhs],
            mut out: DisjointSlice<$out>,
            a_tile: &mut CtaATile,
            b_tile: &mut CtaBTile,
            dims: CtaMatmulDims,
        ) {
            let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
                return;
            };
            cta_accumulate_k_loop2!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1]; {
                $stage(
                    a, b_t, a_tile, b_tile, tile, dims, k_base,
                );
            });
            cta_store2!(
                $store,
                tile,
                &mut out,
                dims,
                acc0,
                acc1
            );
        }
    };
}

causal_body!(
    cta_matmul_f32_half_causal_body,
    u16,
    super::cta_stage_f32::stage_tiles_f32_half_b_t,
    super::cta_store_causal::store_lower
);

causal_body!(@typed cta_matmul_f32_causal_half_out_body, f32, f32, u16,
    super::cta_stage_f32::stage_tiles_f32_b_t, super::cta_store::store_half_lower);

causal_body!(@typed cta_matmul_half_causal_half_out_body, u16, u16, u16,
    super::cta_stage::stage_tiles, super::cta_store::store_half_lower);
