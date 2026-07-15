use cuda_device::DisjointSlice;

use super::cta_tile::CtaMatmulDims;
use super::{CtaATile, CtaBTile};

pub(super) fn cta_matmul_f32_accumulate_body(
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
        super::cta_stage_f32::stage_tiles_f32_b_t(a, b_t, a_tile, b_tile, tile, dims, k_base);
    });
    cta_store4!(
        super::cta_store_accumulate::store,
        tile,
        &mut out,
        dims,
        acc0,
        acc1,
        acc2,
        acc3
    );
}
