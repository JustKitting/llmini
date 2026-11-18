//! Complete paired bounded-SwiGLU reverse graph. Dense products stay NVFP4.
use super::args::{MlpBackwardArgs, MlpBackwardGrads, MlpBackwardScratch};
use crate::backward::linear::{
    run_rowwise_linear_backward, run_rowwise_linear_backward_routed, RowwiseLinearBackwardPass,
};
use crate::{
    uses_block_topk_mlp, MlpWeights, GPT2_EMBEDDING_DIM, GPT2_MLP_DIM,
    GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES, GPT2_MLP_ROUTE_FEATURE_TILES, GPT2_MLP_UP_DIM,
};
use cuda_core::DriverError;
use rust_kernels_cuda::linear_backward::LinearBackwardRoute;

pub(super) fn backward(args: MlpBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let MlpBackwardArgs {
        block_index,
        stream,
        modules,
        saved,
        projections,
        d_residual_out,
        precomputed_d_residual_amax_chunks,
        grads,
        scratch,
        seeds,
    } = args;
    let MlpBackwardScratch {
        down_linear,
        up_linear,
        paired,
        tma_descriptors,
        projection_observer,
    } = scratch;
    let MlpBackwardGrads {
        d_mlp_up,
        d_ln_2_normalized,
        d_c_proj_weight,
        d_c_proj_bias,
        d_c_fc_weight,
        d_c_fc_bias,
    } = grads;
    let down = RowwiseLinearBackwardPass {
        e: d_residual_out,
        saved_input: saved.mlp_down_input_nvfp4,
        weight: projections.down.weight,
        scratch: down_linear,
        dinput: d_mlp_up,
        dweight: d_c_proj_weight,
        dbias: d_c_proj_bias,
        row_count: saved.row_count,
        input_dim: GPT2_MLP_DIM,
        output_dim: GPT2_EMBEDDING_DIM,
        sign_seed: seeds.down_sign,
        scale_seed: seeds.down_scale,
        precomputed_e_amax_chunks: precomputed_d_residual_amax_chunks,
    };
    let routed = uses_block_topk_mlp(block_index);
    if routed {
        run_rowwise_linear_backward_routed(
            modules.linear,
            modules.quant,
            stream,
            down,
            LinearBackwardRoute::mlp_down(
                saved.mlp_route_masks,
                saved.row_count / crate::GPT2_MLP_ROUTE_TILE as u32,
                GPT2_MLP_ROUTE_FEATURE_TILES as u32,
                GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES as u32,
            ),
        )?;
    } else {
        run_rowwise_linear_backward(modules.linear, modules.quant, stream, down)?;
    }
    // Weights and saved input are unchanged since forward. Recreate the same
    // FP32 paired affine accumulator without a raw tape or stochastic draw.
    // The activation kernel rounds both raw inputs to FP16 exactly as forward;
    // this workspace never supplies higher-precision nonlinear inputs.
    MlpWeights::project_paired(
        modules.tma,
        stream,
        saved.mlp_up_input_nvfp4,
        projections.up,
        tma_descriptors,
        paired,
        saved.row_count,
    )?;
    if let Some(observer) = projection_observer {
        observer.observe(
            block_index,
            stream,
            paired,
            saved.row_count as usize * crate::GPT2_MLP_UP,
            true,
        )?;
    }
    modules.mlp.bounded_backward(
        stream,
        paired,
        d_mlp_up,
        saved.mlp_route_masks,
        saved.row_count,
        GPT2_MLP_DIM,
        routed,
    )?;
    // Both gate/up halves are explicitly masked before native backward.
    // Dense evaluation of these zeros preserves the exact routed graph; do
    // not index the paired86 output tiles using a43-tile route descriptor.
    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
            e: paired,
            saved_input: saved.mlp_up_input_nvfp4,
            weight: projections.up.weight,
            scratch: up_linear,
            dinput: d_ln_2_normalized,
            dweight: d_c_fc_weight,
            dbias: d_c_fc_bias,
            row_count: saved.row_count,
            input_dim: GPT2_EMBEDDING_DIM,
            output_dim: GPT2_MLP_UP_DIM,
            sign_seed: seeds.up_sign,
            scale_seed: seeds.up_scale,
            precomputed_e_amax_chunks: None,
        },
    )
}
