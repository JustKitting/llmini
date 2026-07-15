use cuda_core::DriverError;
use rust_kernels_cuda::mlp::Relu2BackwardF16Args;

use super::args::{MlpBackwardArgs, MlpBackwardGrads, MlpBackwardScratch};
use crate::backward::linear::{RowwiseLinearBackwardPass, run_rowwise_linear_backward};
use crate::{GPT2_EMBEDDING_DIM, GPT2_MLP_DIM};

pub fn backward(args: MlpBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let MlpBackwardArgs {
        stream,
        modules,
        saved,
        projections,
        d_residual_out,
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
        d_mlp_relu2,
        d_mlp_up,
        d_ln_2_normalized,
        d_c_proj_weight,
        d_c_proj_bias,
        d_c_fc_weight,
        d_c_fc_bias,
    } = grads;

    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
            e: d_residual_out,
            saved_input: saved.mlp_down_input_nvfp4,
            weight: projections.down.weight,
            scratch: down_linear,
            dinput: d_mlp_relu2,
            dweight: d_c_proj_weight,
            dbias: d_c_proj_bias,
            row_count: saved.row_count,
            input_dim: GPT2_MLP_DIM,
            output_dim: GPT2_EMBEDDING_DIM,
            sign_seed: seeds.down_sign,
            scale_seed: seeds.down_scale,
            precomputed_e_amax_chunks: None,
        },
    )?;

    let d_mlp_up_amax_chunks = modules.mlp.relu2_backward_f16(Relu2BackwardF16Args {
        stream,
        pre_activation: saved.mlp_up,
        d_out: d_mlp_relu2,
        d_pre_activation: d_mlp_up,
        d_pre_activation_chunk_amax: &mut *up_linear.e_h.chunk_amax,
        len: saved.row_count * GPT2_MLP_DIM,
    })?;

    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
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
        },
    )
}
