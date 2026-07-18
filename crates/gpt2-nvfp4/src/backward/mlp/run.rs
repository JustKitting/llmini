use super::args::{MlpBackwardArgs, MlpBackwardGrads, MlpBackwardScratch};
use crate::backward::linear::{
    RowwiseLinearBackwardPass, run_rowwise_linear_backward,
    run_rowwise_linear_backward_relu2_backward_f16,
    run_rowwise_linear_backward_relu2_backward_f16_routed, run_rowwise_linear_backward_routed,
};
use crate::{
    GPT2_EMBEDDING_DIM, GPT2_MLP_DIM, GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES,
    GPT2_MLP_ROUTE_FEATURE_TILES, GPT2_MLP_ROUTE_TOKEN_TILES, uses_block_topk_mlp,
};
use cuda_core::DriverError;
use rust_kernels_cuda::linear_backward::LinearBackwardRoute;

pub fn backward(args: MlpBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
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
        ..
    } = scratch;
    let up_linear = up_linear;
    let MlpBackwardGrads {
        d_mlp_up,
        d_ln_2_normalized,
        d_c_proj_weight,
        d_c_proj_bias,
        d_c_fc_weight,
        d_c_fc_bias,
    } = grads;

    let down_pass = RowwiseLinearBackwardPass {
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
    let d_mlp_up_amax_chunks = if uses_block_topk_mlp(block_index) {
        run_rowwise_linear_backward_relu2_backward_f16_routed(
            modules.linear,
            modules.quant,
            stream,
            down_pass,
            saved.mlp_up,
            &mut *up_linear.e_h.chunk_amax,
            LinearBackwardRoute::mlp_down(
                saved.mlp_route_masks,
                GPT2_MLP_ROUTE_TOKEN_TILES as u32,
                GPT2_MLP_ROUTE_FEATURE_TILES as u32,
                GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES as u32,
            ),
        )?
    } else {
        run_rowwise_linear_backward_relu2_backward_f16(
            modules.linear,
            modules.quant,
            stream,
            down_pass,
            saved.mlp_up,
            &mut *up_linear.e_h.chunk_amax,
        )?
    };

    let up_pass = RowwiseLinearBackwardPass {
        e: d_mlp_up,
        saved_input: saved.mlp_up_input_nvfp4,
        weight: projections.up.weight,
        scratch: up_linear,
        dinput: d_ln_2_normalized,
        dweight: d_c_fc_weight,
        dbias: d_c_fc_bias,
        row_count: saved.row_count,
        input_dim: GPT2_EMBEDDING_DIM,
        output_dim: GPT2_MLP_DIM,
        sign_seed: seeds.up_sign,
        scale_seed: seeds.up_scale,
        precomputed_e_amax_chunks: Some(d_mlp_up_amax_chunks),
    };
    if uses_block_topk_mlp(block_index) {
        run_rowwise_linear_backward_routed(
            modules.linear,
            modules.quant,
            stream,
            up_pass,
            LinearBackwardRoute::mlp_up(
                saved.mlp_route_masks,
                GPT2_MLP_ROUTE_TOKEN_TILES as u32,
                GPT2_MLP_ROUTE_FEATURE_TILES as u32,
                GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES as u32,
            ),
        )
    } else {
        run_rowwise_linear_backward(modules.linear, modules.quant, stream, up_pass)
    }
}
