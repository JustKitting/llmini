use cuda_device::{DisjointSlice, cuda_module, kernel};

use super::convert::fp32_to_f16_body;
use super::cta::{cta_matmul_body, cta_matmul_lower_body};
use super::cta_add_f32::cta_matmul_add_f32_body;
use super::cta_add_f32_rhs_transposed_base::cta_matmul_add_f32_rhs_transposed_base_body;
use super::cta_f32::{
    cta_matmul_f32_body, cta_matmul_f32_lower_body, cta_matmul_f32_windowed_lower_body,
};
use super::cta_f32_a_transposed_half_rhs::{
    cta_matmul_f32_a_transposed_half_rhs_body, cta_matmul_f32_a_transposed_half_rhs_lower_a_body,
};
use super::cta_f32_a_transposed_rhs::cta_matmul_f32_a_transposed_rhs_body;
use super::cta_f32_accumulate::cta_matmul_f32_accumulate_body;
use super::cta_f32_causal::{
    cta_matmul_f32_a_transposed_rhs_strict_neg_body, cta_matmul_f32_causal_body,
    cta_matmul_f32_strict_causal_body,
};
use super::cta_f32_half_rhs::{cta_matmul_f32_half_rhs_body, cta_matmul_f32_half_rhs_lower_a_body};
use super::cta_f32_rhs::cta_matmul_f32_rhs_body;
use super::cta_half_rhs::{
    cta_matmul_half_a_transposed_rhs_lower_a_body,
    cta_matmul_half_a_transposed_rhs_windowed_lower_a_body,
    cta_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_body,
    cta_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled_body,
    cta_matmul_half_rhs_lower_a_body, cta_matmul_half_rhs_windowed_lower_a_body,
    cta_matmul_half_rhs_windowed_lower_a_sparse_body,
};
use super::cta_lower_ds::{
    cta_matmul_lower_ds_body, cta_matmul_windowed_lower_ds_body,
    cta_matmul_windowed_lower_ds_sparse_body,
};
use super::pad::pad_rows_body;

pub const F16_THREADS_PER_BLOCK: u32 = 256;

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[cuda_module]
pub(super) mod module {
    use super::*;

    macro_rules! call_with_tiles {
        ($body:ident; $($pre:expr),* ; $batch_count:expr, $m:expr, $n:expr, $k:expr $(, $extra:expr)* $(,)?) => {{
            static mut A_TILE: crate::f16_tc_matmul::CtaATile = crate::f16_tc_matmul::CtaATile::UNINIT;
            static mut B_TILE: crate::f16_tc_matmul::CtaBTile = crate::f16_tc_matmul::CtaBTile::UNINIT;
            let dims = crate::f16_tc_matmul::cta_tile::CtaMatmulDims::new($batch_count, $m, $n, $k);
            $body($($pre,)* unsafe { &mut A_TILE }, unsafe { &mut B_TILE }, dims $(, $extra)*);
        }};
    }

    #[kernel]
    pub fn f16_fp32_pad_rows_kernel(
        src: &[f32],
        dst: DisjointSlice<f32>,
        rows: u32,
        src_cols: u32,
        dst_cols: u32,
    ) {
        pad_rows_body(src, dst, rows, src_cols, dst_cols);
    }

    #[kernel]
    pub fn fp32_to_f16_kernel(src: &[f32], dst: DisjointSlice<u16>, element_count: u32) {
        fp32_to_f16_body(src, dst, element_count);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_kernel(
        a: &[u16],
        b_t: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_lower_kernel(
        a: &[u16],
        b_t: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_lower_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_lower_ds_kernel(
        a: &[u16],
        b_t: &[u16],
        probs: &[u16],
        softmax_d: &[f32],
        out: DisjointSlice<u16>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(
            cta_matmul_lower_ds_body; a, b_t, probs, softmax_d, out;
            batch_count, m, n, k
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_windowed_lower_ds_kernel(
        a: &[u16],
        b_t: &[u16],
        probs: &[u16],
        softmax_d: &[f32],
        out: DisjointSlice<u16>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_windowed_lower_ds_body; a, b_t, probs, softmax_d, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_windowed_lower_ds_sparse_kernel(
        a: &[u16],
        b_t: &[u16],
        probs: &[u16],
        softmax_d: &[f32],
        tile_scales: &[f32],
        out: DisjointSlice<u16>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_windowed_lower_ds_sparse_body;
            a, b_t, probs, softmax_d, tile_scales, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_rhs_lower_a_kernel(
        a: &[u16],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_half_rhs_lower_a_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_a_transposed_rhs_lower_a_kernel(
        a: &[u16],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_a_transposed_rhs_lower_a_body; a, rhs, out;
            batch_count, m, n, k
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_rhs_windowed_lower_a_kernel(
        a: &[u16],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_rhs_windowed_lower_a_body; a, rhs, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_kernel(
        a: &[u16],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_a_transposed_rhs_windowed_lower_a_body; a, rhs, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_rhs_windowed_lower_a_sparse_kernel(
        a: &[u16],
        rhs: &[u16],
        tile_scales: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_rhs_windowed_lower_a_sparse_body;
            a, rhs, tile_scales, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_kernel(
        a: &[u16],
        rhs: &[u16],
        tile_scales: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_body;
            a, rhs, tile_scales, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled_kernel(
        a: &[u16],
        rhs: &[u16],
        tile_scales: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_half_a_transposed_rhs_windowed_lower_a_sparse_scaled_body;
            a, rhs, tile_scales, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_accumulate_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_accumulate_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_causal_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_causal_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_strict_causal_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_strict_causal_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_lower_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_lower_body; a, b_t, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_windowed_lower_kernel(
        a: &[f32],
        b_t: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        window: u32,
    ) {
        call_with_tiles!(
            cta_matmul_f32_windowed_lower_body; a, b_t, out;
            batch_count, m, n, k, window
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_rhs_kernel(
        a: &[f32],
        rhs: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_rhs_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_half_rhs_kernel(
        a: &[f32],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_half_rhs_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_half_rhs_lower_a_kernel(
        a: &[f32],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_half_rhs_lower_a_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_a_transposed_rhs_kernel(
        a: &[f32],
        rhs: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_a_transposed_rhs_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_a_transposed_rhs_strict_neg_kernel(
        a: &[f32],
        rhs: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_a_transposed_rhs_strict_neg_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_a_transposed_half_rhs_kernel(
        a: &[f32],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(cta_matmul_f32_a_transposed_half_rhs_body; a, rhs, out; batch_count, m, n, k);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_f32_a_transposed_half_rhs_lower_a_kernel(
        a: &[f32],
        rhs: &[u16],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
    ) {
        call_with_tiles!(
            cta_matmul_f32_a_transposed_half_rhs_lower_a_body; a, rhs, out;
            batch_count, m, n, k
        );
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_add_f32_kernel(
        a: &[f32],
        b_t: &[f32],
        base: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        base_scale: f32,
        matmul_scale: f32,
    ) {
        call_with_tiles!(cta_matmul_add_f32_body; a, b_t, base, out; batch_count, m, n, k, base_scale, matmul_scale);
    }

    #[kernel]
    pub fn f16_cta_tc_matmul_add_f32_rhs_transposed_base_kernel(
        a: &[f32],
        rhs: &[f32],
        base: &[f32],
        out: DisjointSlice<f32>,
        batch_count: u32,
        m: u32,
        n: u32,
        k: u32,
        base_scale: f32,
        matmul_scale: f32,
    ) {
        call_with_tiles!(
            cta_matmul_add_f32_rhs_transposed_base_body; a, rhs, base, out;
            batch_count, m, n, k, base_scale, matmul_scale
        );
    }
}

pub(crate) use module::LoadedModule;
