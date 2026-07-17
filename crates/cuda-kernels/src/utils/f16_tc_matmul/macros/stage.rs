macro_rules! cta_stage_transposed_rhs_fn {
    ($name:ident, $rhs_ty:ty, $load:path, |$lo:ident, $hi:ident| $packed:expr) => {
        fn $name(
            rhs: &[$rhs_ty],
            b_tile: &mut $crate::f16_tc_matmul::CtaBTile,
            tile: $crate::f16_tc_matmul::cta_tile::CtaTile,
            n: u32,
            k: u32,
            k_base: u32,
        ) {
            let mut pair = cuda_device::thread::threadIdx_x() * 2;
            while pair < $crate::f16_tc_matmul::cta_tile::CTA_B_ELEMS as u32 {
                let (global_row, global_col) =
                    $crate::f16_tc_matmul::cta_stage::stage_coords(pair, tile.col_base, k_base);
                let $lo: $rhs_ty = if global_row < n && global_col < k {
                    $load(
                        rhs.as_ptr(),
                        ((tile.batch * k + global_col) * n + global_row) as usize,
                    )
                } else {
                    0 as $rhs_ty
                };
                let hi_col = global_col + 1;
                let $hi: $rhs_ty = if global_row < n && hi_col < k {
                    $load(
                        rhs.as_ptr(),
                        ((tile.batch * k + hi_col) * n + global_row) as usize,
                    )
                } else {
                    0 as $rhs_ty
                };
                $crate::f16_tc_matmul::convert::store_f16x2_shared(
                    b_tile.as_mut_ptr(),
                    pair as usize,
                    $packed,
                );
                pair += cuda_device::thread::blockDim_x() * 2;
            }
        }
    };
}
