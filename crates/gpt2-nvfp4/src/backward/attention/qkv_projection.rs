use cuda_core::DriverError;

use super::types::AttentionQkvBackwardArgs;
use crate::AttentionDims;
use crate::backward::linear::{RowwiseLinearBackwardPass, run_rowwise_linear_backward};

pub fn qkv_projection_backward(
    args: AttentionQkvBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let AttentionQkvBackwardArgs {
        use_full_attention,
        stream,
        modules,
        saved,
        projections,
        d_qkv,
        d_ln_1_normalized,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        precomputed_d_qkv_amax_chunks,
        scratch,
        seeds,
    } = args;
    let dims = AttentionDims::new(use_full_attention);

    run_rowwise_linear_backward(
        modules.linear,
        modules.quant,
        stream,
        RowwiseLinearBackwardPass {
            e: d_qkv,
            saved_input: saved.qkv_input_nvfp4,
            weight: projections.qkv_weight,
            scratch: scratch.linear,
            dinput: d_ln_1_normalized,
            dweight: d_attn_qkv_weight,
            dbias: d_attn_qkv_bias,
            row_count: saved.row_count,
            input_dim: dims.embedding_dim,
            output_dim: dims.qkv_dim,
            sign_seed: seeds.sign,
            scale_seed: seeds.scale,
            precomputed_e_amax_chunks: precomputed_d_qkv_amax_chunks,
        },
    )
}
