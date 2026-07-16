macro_rules! cta_bt_matmul_body_fn {
    ($name:ident, $lhs_ty:ty, $rhs_ty:ty, $stage:path, $stage_aligned:path) => {
        pub(super) fn $name(
            a: &[$lhs_ty], b_t: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            dims: $crate::f16_tc_matmul::cta_tile::CtaMatmulDims,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            let aligned = dims.aligned();
            cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
                if aligned {
                    $stage_aligned(a, b_t, a_tile, b_tile, tile, dims, k_base);
                } else {
                    $stage(a, b_t, a_tile, b_tile, tile, dims, k_base);
                }
            });
            if aligned {
                cta_store4!($crate::f16_tc_matmul::cta_store::store_aligned, tile, &mut out, dims, acc0, acc1, acc2, acc3);
            } else {
                cta_store4!($crate::f16_tc_matmul::cta_store::store, tile, &mut out, dims, acc0, acc1, acc2, acc3);
            }
        }
    };
}

macro_rules! cta_bt_matmul_lower_body_fn {
    ($name:ident, $lhs_ty:ty, $rhs_ty:ty, $stage:path, $stage_aligned:path) => {
        pub(super) fn $name(
            a: &[$lhs_ty], b_t: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            dims: $crate::f16_tc_matmul::cta_tile::CtaMatmulDims,
        ) {
            if cuda_device::thread::blockIdx_x() > cuda_device::thread::blockIdx_y() {
                return;
            }
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            let aligned = dims.aligned();
            cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
                if aligned {
                    $stage_aligned(a, b_t, a_tile, b_tile, tile, dims, k_base);
                } else {
                    $stage(a, b_t, a_tile, b_tile, tile, dims, k_base);
                }
            });
            if aligned {
                cta_store4!($crate::f16_tc_matmul::cta_store::store_aligned, tile, &mut out, dims, acc0, acc1, acc2, acc3);
            } else {
                cta_store4!($crate::f16_tc_matmul::cta_store::store, tile, &mut out, dims, acc0, acc1, acc2, acc3);
            }
        }
    };
}

macro_rules! cta_rhs_matmul_body_fn {
    ($name:ident, $rhs:ident: $rhs_ty:ty, $stage:path) => {
        pub(super) fn $name(
            a: &[f32], $rhs: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            dims: $crate::f16_tc_matmul::cta_tile::CtaMatmulDims,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
                $stage(a, $rhs, a_tile, b_tile, tile, dims, k_base);
            });
            cta_store4!($crate::f16_tc_matmul::cta_store::store, tile, &mut out, dims, acc0, acc1, acc2, acc3);
        }
    };
}

macro_rules! cta_rhs_matmul_k_major_a_body_fn {
    ($name:ident, $rhs:ident: $rhs_ty:ty, $stage:path) => {
        pub(super) fn $name(
            a: &[f32],
            $rhs: &[$rhs_ty],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            dims: $crate::f16_tc_matmul::cta_tile::CtaMatmulDims,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            cta_accumulators!(acc0, acc1, acc2, acc3);
            let mut k_base = 0;
            while k_base < dims.k {
                $stage(a, $rhs, a_tile, b_tile, tile, dims, k_base);
                cuda_device::thread::sync_threads();
                cta_mma4_k_major_a!(a_tile, b_tile, tile, acc0, acc1, acc2, acc3);
                $crate::f16_tc_matmul::cta_sync::sync_before_next_k(k_base, dims.k);
                k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
            }
            cta_store4!(
                $crate::f16_tc_matmul::cta_store::store,
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

macro_rules! cta_add_matmul_body_fn {
    ($name:ident, $rhs:ident, $stage:path) => {
        #[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
        pub(super) fn $name(
            a: &[f32], $rhs: &[f32], base: &[f32],
            mut out: cuda_device::DisjointSlice<f32>,
            a_tile: &mut $crate::f16_tc_matmul::CtaATile,
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            dims: $crate::f16_tc_matmul::cta_tile::CtaMatmulDims,
            base_scale: f32, matmul_scale: f32,
        ) {
            let Some(tile) = $crate::f16_tc_matmul::cta_tile::active_tile(dims.batch_count) else {
                return;
            };
            cta_accumulate_k_loop4!(tile, a_tile, b_tile, dims.k, k_base, [acc0, acc1, acc2, acc3]; {
                $stage(a, $rhs, a_tile, b_tile, tile, dims, k_base);
            });
            cta_store_add4!(
                $crate::f16_tc_matmul::cta_store_add::store_add, tile, base, &mut out, dims, base_scale,
                matmul_scale, acc0 => 0, acc1 => 1, acc2 => 2, acc3 => 3,
            );
        }
    };
}
