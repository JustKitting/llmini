cta_bt_matmul_body_fn!(
    cta_matmul_f32_body,
    f32,
    f32,
    super::cta_stage_f32::stage_tiles_f32_b_t,
    super::cta_stage_f32::stage_tiles_f32_b_t_aligned
);

cta_bt_matmul_lower_body_fn!(
    cta_matmul_f32_lower_body,
    f32,
    f32,
    super::cta_stage_f32::stage_tiles_f32_b_t,
    super::cta_stage_f32::stage_tiles_f32_b_t_aligned
);

pub(super) fn cta_matmul_f32_windowed_lower_body(
    a: &[f32],
    b_t: &[f32],
    mut out: cuda_device::DisjointSlice<f32>,
    a_tile: &mut super::CtaATile,
    b_tile: &mut super::CtaBTile,
    dims: super::cta_tile::CtaMatmulDims,
    window: u32,
) {
    let tile_col = cuda_device::thread::blockIdx_x();
    let tile_row = cuda_device::thread::blockIdx_y();
    let window_tiles = window / super::cta_tile::CTA_N;
    if tile_col > tile_row || tile_row > tile_col + window_tiles {
        return;
    }
    let Some(tile) = super::cta_tile::active_wide_tile(dims.batch_count) else {
        return;
    };
    let aligned = dims.aligned();
    cta_accumulate_k_loop2!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1]; {
        if aligned {
            super::cta_stage_f32::stage_tiles_f32_b_t_aligned(
                a, b_t, a_tile, b_tile, tile, dims, k_base,
            );
        } else {
            super::cta_stage_f32::stage_tiles_f32_b_t(
                a, b_t, a_tile, b_tile, tile, dims, k_base,
            );
        }
    });
    if aligned {
        cta_store2!(
            super::cta_store::store_aligned,
            tile,
            &mut out,
            dims,
            acc0,
            acc1
        );
    } else {
        cta_store2!(super::cta_store::store, tile, &mut out, dims, acc0, acc1);
    }
}
