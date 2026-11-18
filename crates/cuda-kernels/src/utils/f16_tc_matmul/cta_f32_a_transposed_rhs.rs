cta_rhs_matmul_body_fn!(
    cta_matmul_f32_a_transposed_rhs_body,
    rhs: f32,
    super::cta_stage_f32_transposed::stage_tiles_f32_a_transposed_rhs
);

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
pub(super) fn cta_matmul_half_a_transposed_dual_rhs_body(
    a: &[u16],
    rhs0: &[u16],
    rhs1: &[u16],
    mut out0: cuda_device::DisjointSlice<f32>,
    mut out1: cuda_device::DisjointSlice<f32>,
    a_tile: &mut super::CtaATile,
    b_tiles: (&mut super::CtaBTile, &mut super::CtaBTile),
    dims: super::cta_tile::CtaMatmulDims,
) {
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    let (b0, b1) = b_tiles;
    cta_accumulators!(acc00, acc01, acc10, acc11);
    let mut k_base = 0;
    while k_base < dims.k {
        super::cta_stage_f32_transposed::stage_half_a_transposed(a, a_tile, tile, dims, k_base);
        super::cta_stage_f32_transposed::stage_half_rhs_only(rhs0, b0, tile, dims, k_base);
        super::cta_stage_f32_transposed::stage_half_rhs_only(rhs1, b1, tile, dims, k_base);
        cuda_device::thread::sync_threads();
        cta_mma2!(a_tile, b0, tile, acc00, acc01);
        cta_mma2!(a_tile, b1, tile, acc10, acc11);
        super::cta_sync::sync_before_next_k(k_base, dims.k);
        k_base += super::cta_tile::CTA_K;
    }
    cta_store2!(super::cta_store::store, tile, &mut out0, dims, acc00, acc01);
    cta_store2!(super::cta_store::store, tile, &mut out1, dims, acc10, acc11);
}

cta_rhs_matmul_body_fn!(@typed cta_matmul_half_a_transposed_f32_rhs_body, u16, rhs:f32, f32,
    super::cta_stage_f32_transposed::stage_tiles_half_a_transposed_f32_rhs, super::cta_store::store);
