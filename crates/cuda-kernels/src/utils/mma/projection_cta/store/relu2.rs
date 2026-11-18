use cuda_device::DisjointSlice;

use crate::float_ptx::max_f32;
use crate::mma::projection::Nvfp4ProjectionParams;

use super::super::tile::Nvfp4ProjectionCtaTile;
use super::common::{affine_pair_scaled, affine_value, aligned_row_col, row_col};

struct Relu2StoreArgs<'a, 'pre, 'out> {
    input_global_scales: &'a [f32],
    bias_bytes: &'a [u8],
    bias_scales: &'a [u8],
    pre_activation: &'a mut DisjointSlice<'pre, f32>,
    out: &'a mut DisjointSlice<'out, f32>,
    params: &'a Nvfp4ProjectionParams,
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn store_relu2_accumulator(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let mut args = Relu2StoreArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        pre_activation,
        out,
        params,
    };
    store_acc4!(store_one, acc, tile, &mut args);
}

#[expect(clippy::too_many_arguments, reason = "CUDA ABI uses explicit buffers")]
#[inline(always)]
pub fn store_relu2_accumulator_aligned(
    acc: [f32; 4],
    input_global_scales: &[f32],
    bias_bytes: &[u8],
    bias_scales: &[u8],
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
    tile: Nvfp4ProjectionCtaTile,
    params: &Nvfp4ProjectionParams,
) {
    let mut args = Relu2StoreArgs {
        input_global_scales,
        bias_bytes,
        bias_scales,
        pre_activation,
        out,
        params,
    };
    let (row0, row1, col0) = aligned_row_col(tile);
    store_pair_aligned(acc[0], acc[1], row0, col0, &mut args);
    store_pair_aligned(acc[2], acc[3], row1, col0, &mut args);
}

#[inline(always)]
fn store_one(
    acc: f32,
    index: u32,
    tile: Nvfp4ProjectionCtaTile,
    args: &mut Relu2StoreArgs<'_, '_, '_>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let pre = affine_value(
            acc,
            row,
            col,
            args.input_global_scales,
            args.bias_bytes,
            args.bias_scales,
            args.params,
        );
        let relu = max_f32(pre, 0.0);
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *args.pre_activation.get_unchecked_mut(offset) = pre;
            *args.out.get_unchecked_mut(offset) = crate::activation::centered_relu2_from_positive(relu);
        }
    }
}

#[inline(always)]
fn store_pair_aligned(
    acc0: f32,
    acc1: f32,
    row: u32,
    col0: u32,
    args: &mut Relu2StoreArgs<'_, '_, '_>,
) {
    if row >= args.params.token_count || col0 + 1 >= args.params.output_dim {
        return;
    }
    let scale = args.input_global_scales[row as usize] * args.params.weight_global_scale;
    let (pre0, pre1) = affine_pair_scaled(
        (acc0, acc1),
        scale,
        col0,
        (args.bias_bytes, args.bias_scales),
        args.params,
    );
    let relu0 = max_f32(pre0, 0.0);
    let relu1 = max_f32(pre1, 0.0);
    let offset = row as usize * args.params.output_dim as usize + col0 as usize;
    unsafe {
        *args.pre_activation.get_unchecked_mut(offset) = pre0;
        *args.pre_activation.get_unchecked_mut(offset + 1) = pre1;
        *args.out.get_unchecked_mut(offset) = crate::activation::centered_relu2_from_positive(relu0);
        *args.out.get_unchecked_mut(offset + 1) = crate::activation::centered_relu2_from_positive(relu1);
    }
}
