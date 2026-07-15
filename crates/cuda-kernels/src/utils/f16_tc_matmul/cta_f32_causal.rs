use cuda_device::DisjointSlice;

use super::cta_tile::CtaMatmulDims;
use super::{CtaATile, CtaBTile};

macro_rules! causal_body {
    ($name:ident, $store:path) => {
        pub(super) fn $name(
            a: &[f32],
            b_t: &[f32],
            mut out: DisjointSlice<f32>,
            a_tile: &mut CtaATile,
            b_tile: &mut CtaBTile,
            dims: CtaMatmulDims,
        ) {
            let Some(tile) = super::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
                super::cta_stage_f32::stage_tiles_f32_b_t(
                    a, b_t, a_tile, b_tile, tile, dims, k_base,
                );
            });
            cta_store4!(
                $store,
                tile,
                &mut out,
                dims,
                acc0,
                acc1,
                acc2,
                acc3
            );
        }
    };
}

causal_body!(
    cta_matmul_f32_causal_body,
    super::cta_store_causal::store_lower
);
causal_body!(
    cta_matmul_f32_strict_causal_body,
    super::cta_store_causal::store_strict_lower
);

pub(super) fn cta_matmul_f32_a_transposed_rhs_strict_neg_body(
    a: &[f32],
    rhs: &[f32],
    mut out: DisjointSlice<f32>,
    a_tile: &mut CtaATile,
    b_tile: &mut CtaBTile,
    dims: CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_tile(dims.batch_count) else {
        return;
    };
    cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
        super::cta_stage_f32_transposed::stage_tiles_f32_a_transposed_rhs(
            a, rhs, a_tile, b_tile, tile, dims, k_base,
        );
    });
    cta_store4!(
        super::cta_store_causal::store_strict_neg,
        tile,
        &mut out,
        dims,
        acc0,
        acc1,
        acc2,
        acc3
    );
}
