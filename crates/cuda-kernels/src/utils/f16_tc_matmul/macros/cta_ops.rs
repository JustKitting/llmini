macro_rules! cta_accumulators {
    ($($acc:ident),+ $(,)?) => {
        $(let mut $acc = [0.0_f32; 4];)+
    };
}

macro_rules! cta_mma4 {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {
        cta_mma4!($a_tile, $b_tile, $tile, $acc0 => 0, $acc1 => 1, $acc2 => 2, $acc3 => 3)
    };
    ($a_tile:expr, $b_tile:expr, $tile:expr, $($acc:ident => $offset:expr),+ $(,)?) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments($a_tile, tile);
        $(
            $crate::mma::mma_m16n8k16_f16_f16_f32(
                a_fragments,
                $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + $offset),
                &mut $acc,
            );
        )+
    }};
}

macro_rules! cta_mma2 {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident) => {
        cta_mma4!($a_tile, $b_tile, $tile, $acc0 => 0, $acc1 => 1)
    };
}

macro_rules! cta_mma2_k_major_b {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments($a_tile, tile);
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments_k_major($b_tile, tile, tile.warp_n0),
            &mut $acc0,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments_k_major(
                $b_tile,
                tile,
                tile.warp_n0 + 1,
            ),
            &mut $acc1,
        );
    }};
}

macro_rules! cta_mma2_k_major_a {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments_k_major($a_tile, tile);
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0),
            &mut $acc0,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments($b_tile, tile, tile.warp_n0 + 1),
            &mut $acc1,
        );
    }};
}

macro_rules! cta_mma2_k_major_ab {
    ($a_tile:expr, $b_tile:expr, $tile:expr, $acc0:ident, $acc1:ident) => {{
        let tile = $tile;
        let a_fragments = $crate::f16_tc_matmul::cta_stage::load_a_fragments_k_major($a_tile, tile);
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments_k_major($b_tile, tile, tile.warp_n0),
            &mut $acc0,
        );
        $crate::mma::mma_m16n8k16_f16_f16_f32(
            a_fragments,
            $crate::f16_tc_matmul::cta_stage::load_b_fragments_k_major(
                $b_tile,
                tile,
                tile.warp_n0 + 1,
            ),
            &mut $acc1,
        );
    }};
}

macro_rules! cta_accumulate_k_loop2 {
    ($tile:expr, $a_tile:expr, $b_tile:expr, $k:expr, $k_base:ident,
     [$acc0:ident, $acc1:ident]; $stage:block) => {
        cta_accumulators!($acc0, $acc1);
        let mut $k_base = 0;
        while $k_base < $k {
            $stage
            cuda_device::thread::sync_threads();
            cta_mma2!($a_tile, $b_tile, $tile, $acc0, $acc1);
            $crate::f16_tc_matmul::cta_sync::sync_before_next_k($k_base, $k);
            $k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
        }
    };
}

macro_rules! cta_accumulate_k_loop4 {
    ($tile:expr, $a_tile:expr, $b_tile:expr, $k:expr, $k_base:ident,
     [$acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident]; $stage:block) => {
        cta_accumulators!($acc0, $acc1, $acc2, $acc3);
        let mut $k_base = 0;
        while $k_base < $k {
            $stage
            cuda_device::thread::sync_threads();
            cta_mma4!($a_tile, $b_tile, $tile, $acc0, $acc1, $acc2, $acc3);
            $crate::f16_tc_matmul::cta_sync::sync_before_next_k($k_base, $k);
            $k_base += $crate::f16_tc_matmul::cta_tile::CTA_K;
        }
    };
}

macro_rules! cta_store2 {
    ($store:path, $tile:expr, $out:expr, $dims:expr, $acc0:ident, $acc1:ident) => {{
        let tile = $tile;
        $store($acc0, tile, tile.warp_n0, $out, $dims.m, $dims.n);
        $store($acc1, tile, tile.warp_n0 + 1, $out, $dims.m, $dims.n);
    }};
}

macro_rules! cta_store4 {
    ($store:path, $tile:expr, $out:expr, $dims:expr, $acc0:ident, $acc1:ident, $acc2:ident, $acc3:ident) => {{
        let tile = $tile;
        $store($acc0, tile, tile.warp_n0, $out, $dims.m, $dims.n);
        $store($acc1, tile, tile.warp_n0 + 1, $out, $dims.m, $dims.n);
        $store($acc2, tile, tile.warp_n0 + 2, $out, $dims.m, $dims.n);
        $store($acc3, tile, tile.warp_n0 + 3, $out, $dims.m, $dims.n);
    }};
}

macro_rules! cta_store_add4 {
    (
        $store_add:path, $tile:expr, $base:expr, $out:expr, $dims:expr, $base_scale:expr,
        $matmul_scale:expr, $($acc:ident => $offset:expr),+ $(,)?
    ) => {{
        let tile = $tile;
        $(
            $store_add(
                $acc,
                $out,
                $crate::f16_tc_matmul::cta_store_add::StoreAddArgs::new(
                    tile,
                    tile.warp_n0 + $offset,
                    $base,
                    $dims,
                    $base_scale,
                    $matmul_scale,
                ),
            );
        )+
    }};
}
