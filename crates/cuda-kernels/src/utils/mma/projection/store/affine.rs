use cuda_device::DisjointSlice;

use crate::float_ptx::max_f32;

use super::{StoreAccumulatorArgs, affine_value, row_col};
use crate::mma::Nvfp4ProjectionTile;

#[inline(always)]
pub(crate) fn store_accumulator(
    acc: [f32; 4],
    tile: Nvfp4ProjectionTile,
    args: StoreAccumulatorArgs<'_>,
    out: &mut DisjointSlice<'_, f32>,
) {
    store_acc4!(store_one, acc, tile, &args, out);
}

#[inline(always)]
pub(crate) fn store_relu2_accumulator(
    acc: [f32; 4],
    tile: Nvfp4ProjectionTile,
    args: StoreAccumulatorArgs<'_>,
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
) {
    store_acc4!(store_relu2_one, acc, tile, &args, pre_activation, out);
}

#[inline(always)]
pub(crate) fn store_residual_accumulator(
    acc: [f32; 4],
    tile: Nvfp4ProjectionTile,
    args: StoreAccumulatorArgs<'_>,
    residual: &mut DisjointSlice<'_, f32>,
    projection_out: &mut DisjointSlice<'_, f32>,
) {
    store_acc4!(
        store_residual_one,
        acc,
        tile,
        &args,
        residual,
        projection_out
    );
}

#[inline(always)]
fn store_one(
    acc: f32,
    index: usize,
    tile: Nvfp4ProjectionTile,
    args: &StoreAccumulatorArgs<'_>,
    out: &mut DisjointSlice<'_, f32>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let value = affine_value(acc, row, col, args);
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *out.get_unchecked_mut(offset) = if args.params.residual_add == 0 {
                value
            } else {
                *out.get_unchecked_mut(offset) + value
            };
        }
    }
}

#[inline(always)]
fn store_relu2_one(
    acc: f32,
    index: usize,
    tile: Nvfp4ProjectionTile,
    args: &StoreAccumulatorArgs<'_>,
    pre_activation: &mut DisjointSlice<'_, f32>,
    out: &mut DisjointSlice<'_, f32>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let pre = affine_value(acc, row, col, args);
        let relu = max_f32(pre, 0.0);
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *pre_activation.get_unchecked_mut(offset) = pre;
            *out.get_unchecked_mut(offset) = crate::activation::centered_relu2_from_positive(relu);
        }
    }
}

#[inline(always)]
fn store_residual_one(
    acc: f32,
    index: usize,
    tile: Nvfp4ProjectionTile,
    args: &StoreAccumulatorArgs<'_>,
    residual: &mut DisjointSlice<'_, f32>,
    projection_out: &mut DisjointSlice<'_, f32>,
) {
    let (row, col) = row_col(tile, index);
    if row < args.params.token_count && col < args.params.output_dim {
        let value = affine_value(acc, row, col, args);
        let offset = row as usize * args.params.output_dim as usize + col as usize;
        unsafe {
            *projection_out.get_unchecked_mut(offset) = value;
            *residual.get_unchecked_mut(offset) += value;
        }
    }
}
