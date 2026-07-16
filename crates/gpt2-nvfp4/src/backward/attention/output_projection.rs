use cuda_core::DriverError;

use super::types::AttentionCProjBackwardArgs;
use crate::GPT2_EMBEDDING_DIM;
use crate::backward::linear::{RowwiseLinearBackwardPass, run_rowwise_linear_backward};

pub fn c_proj_backward(args: AttentionCProjBackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let AttentionCProjBackwardArgs {
        stream,
        modules,
        saved,
        projections,
        d_residual_after_attention,
        precomputed_d_residual_amax_chunks,
        d_attention_out,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        scratch,
        seeds,
    } = args;

    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
            saved_input: saved.c_proj_input_nvfp4,
            weight: projections.c_proj_weight,
            scratch: scratch.linear,
            dinput: d_attention_out,
            dweight: d_attn_c_proj_weight,
            dbias: d_attn_c_proj_bias,
            row_count: saved.row_count,
            input_dim: GPT2_EMBEDDING_DIM,
            output_dim: GPT2_EMBEDDING_DIM,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
            e: d_residual_after_attention,
            precomputed_e_amax_chunks: precomputed_d_residual_amax_chunks,
        },
    )
}
