use cuda_core::DriverError;

use super::types::BlockAttentionBackwardArgs;
use crate::backward::{
    AttentionCProjBackwardArgs, AttentionCoreBackwardArgs, AttentionQkvBackwardArgs,
    Gpt2LayerNormBackwardAddArgs, attention_c_proj_backward, causal_attention_backward,
    layer_norm_backward_add, qkv_projection_backward,
};
use crate::types::BlockBackwardGrads;

pub fn attention_side_backward(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
) -> Result<(), DriverError> {
    let BlockAttentionBackwardArgs {
        block_index,
        use_full_attention,
        reuse_forward_probs,
        stream,
        modules,
        saved,
        ln_1,
        projections,
        d_residual_after_attention,
        d_residual_in,
        d_hidden,
        d_qkv,
        grads,
        scratch,
        seeds,
    } = args;
    let BlockBackwardGrads {
        ln_1: mut ln_1_grads,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        ..
    } = grads;
    let scratch = scratch;
    attention_c_proj_backward(AttentionCProjBackwardArgs {
        stream,
        modules: modules.linear,
        saved,
        projections,
        d_residual_after_attention,
        d_attention_out: &mut *d_hidden,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        scratch: scratch.c_proj,
        seeds: seeds.c_proj,
    })?;
    let d_qkv_amax_chunks = causal_attention_backward(AttentionCoreBackwardArgs {
        block_index,
        use_full_attention,
        reuse_forward_probs,
        stream,
        module: modules.attention,
        tc_module: modules.f16_tc,
        saved,
        d_attention_out: &*d_hidden,
        d_qkv,
        d_qkv_chunk_amax: &mut *scratch.qkv.linear.e_h.chunk_amax,
        scratch: scratch.core,
    })?;
    qkv_projection_backward(AttentionQkvBackwardArgs {
        use_full_attention,
        stream,
        modules: modules.linear,
        saved,
        projections,
        d_qkv: &*d_qkv,
        d_ln_1_normalized: &mut *d_hidden,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        precomputed_d_qkv_amax_chunks: d_qkv_amax_chunks,
        scratch: scratch.qkv,
        seeds: seeds.qkv,
    })?;
    layer_norm_backward_add(Gpt2LayerNormBackwardAddArgs {
        stream,
        module: modules.layer_norm,
        weights: ln_1,
        saved: saved.ln_1,
        grads: ln_1_grads.reborrow(),
        d_normalized: &*d_hidden,
        direct: d_residual_after_attention,
        d_residual: d_residual_in,
    })
}
