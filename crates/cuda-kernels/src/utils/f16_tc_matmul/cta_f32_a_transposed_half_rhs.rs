cta_rhs_matmul_k_major_a_body_fn!(
    cta_matmul_f32_a_transposed_half_rhs_body,
    rhs: u16,
    super::cta_stage_f32_transposed::stage_tiles_f32_a_transposed_half_rhs
);

pub(super) fn cta_matmul_f32_a_transposed_half_rhs_lower_a_body(
    a: &[f32],
    rhs: &[u16],
    mut out: cuda_device::DisjointSlice<f32>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: super::cta_tile::CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_tile(dims.batch_count) else {
        return;
    };
    cta_accumulators!(acc0, acc1, acc2, acc3);
    let mut k_base = tile.row_base;
    while k_base < dims.k {
        super::cta_stage_f32_transposed::stage_tiles_f32_a_transposed_half_rhs_lower_a(
            a, rhs, a_tile, b_tile, tile, dims, k_base,
        );
        cuda_device::thread::sync_threads();
        cta_mma4_k_major_a!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
        super::cta_sync::sync_before_next_k(k_base, dims.k);
        k_base += super::cta_tile::CTA_K;
    }
    cta_store4!(
        super::cta_store::store,
        tile,
        &mut out,
        dims,
        acc0,
        acc1,
        acc2,
        acc3
    );
}
