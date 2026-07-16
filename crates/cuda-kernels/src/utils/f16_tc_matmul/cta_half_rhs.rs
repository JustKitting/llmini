pub(super) fn cta_matmul_half_rhs_lower_a_body(
    a: &[u16],
    rhs: &[u16],
    mut out: cuda_device::DisjointSlice<f32>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: super::cta_tile::CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    cta_accumulators!(acc0, acc1);
    let row_k_limit = tile.row_base + super::cta_tile::CTA_M;
    let k_limit = if row_k_limit < dims.k {
        row_k_limit
    } else {
        dims.k
    };
    let mut k_base = 0;
    while k_base < k_limit {
        super::cta_stage_half::stage_tiles_half_rhs_lower_a(
            a, rhs, a_tile, b_tile, tile, dims, k_base,
        );
        cuda_device::thread::sync_threads();
        cta_mma2!(a_tile, b_tile, tile, acc0, acc1);
        super::cta_sync::sync_before_next_k(k_base, k_limit);
        k_base += super::cta_tile::CTA_K;
    }
    cta_store2!(super::cta_store::store, tile, &mut out, dims, acc0, acc1);
}

pub(super) fn cta_matmul_half_a_transposed_rhs_lower_a_body(
    a: &[u16],
    rhs: &[u16],
    mut out: cuda_device::DisjointSlice<f32>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: super::cta_tile::CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    cta_accumulators!(acc0, acc1);
    let mut k_base = tile.row_base;
    while k_base < dims.k {
        super::cta_stage_half::stage_tiles_half_a_transposed_rhs_lower_a(
            a, rhs, a_tile, b_tile, tile, dims, k_base,
        );
        cuda_device::thread::sync_threads();
        cta_mma2!(a_tile, b_tile, tile, acc0, acc1);
        super::cta_sync::sync_before_next_k(k_base, dims.k);
        k_base += super::cta_tile::CTA_K;
    }
    cta_store2!(super::cta_store::store, tile, &mut out, dims, acc0, acc1);
}
