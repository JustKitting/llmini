use cuda_core::DriverError;
use rust_kernels_cuda::attention::{
    AccumulateValueResidualGradArgs, FinishValueResidualGradArgs, InitializeValueResidualGradArgs,
};

use super::types::BlockAttentionBackwardArgs;
use crate::backward::{
    AttentionCProjBackwardArgs, AttentionCoreBackwardArgs, AttentionQkvBackwardArgs,
    Gpt2LayerNormBackwardAddAmaxArgs, Gpt2LayerNormBackwardAddArgs, attention_c_proj_backward,
    causal_attention_backward, layer_norm_backward_add, layer_norm_backward_add_amax,
    qkv_projection_backward,
};
use crate::types::BlockBackwardGrads;
use crate::{AttentionDims, GPT2_N_LAYER};

pub fn attention_side_backward(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
) -> Result<Option<u32>, DriverError> {
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
        precomputed_d_residual_after_attention_amax_chunks,
        d_residual_in,
        d_residual_in_chunk_amax,
        d_hidden,
        d_qkv,
        d_value_residual,
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
        precomputed_d_residual_amax_chunks: precomputed_d_residual_after_attention_amax_chunks,
        d_attention_out: &mut *d_hidden,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias,
        scratch: scratch.c_proj,
        seeds: seeds.c_proj,
    })?;
    causal_attention_backward(AttentionCoreBackwardArgs {
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
    let dims = AttentionDims::new(use_full_attention);
    if block_index == GPT2_N_LAYER - 1 {
        modules
            .attention
            .initialize_value_residual_grad(InitializeValueResidualGradArgs {
                stream,
                d_qkv: &mut *d_qkv,
                d_first_value: d_value_residual,
                row_count: saved.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
            })?;
    } else if block_index == 0 {
        modules
            .attention
            .finish_value_residual_grad(FinishValueResidualGradArgs {
                stream,
                d_qkv: &mut *d_qkv,
                d_first_value: &*d_value_residual,
                row_count: saved.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
            })?;
    } else {
        modules
            .attention
            .accumulate_value_residual_grad(AccumulateValueResidualGradArgs {
                stream,
                d_qkv: &mut *d_qkv,
                d_first_value: d_value_residual,
                row_count: saved.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
            })?;
    }
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
        // Value-residual routing changes the V section after the attention
        // core computes its amax. Recompute from the transformed gradient.
        precomputed_d_qkv_amax_chunks: None,
        scratch: scratch.qkv,
        seeds: seeds.qkv,
    })?;
    if block_index == 0 {
        layer_norm_backward_add(Gpt2LayerNormBackwardAddArgs {
            stream,
            module: modules.layer_norm,
            weights: ln_1,
            saved: saved.ln_1,
            grads: ln_1_grads.reborrow(),
            d_normalized: &*d_hidden,
            direct: d_residual_after_attention,
            d_residual: d_residual_in,
        })?;
        Ok(None)
    } else {
        let chunk_count = layer_norm_backward_add_amax(Gpt2LayerNormBackwardAddAmaxArgs {
            stream,
            module: modules.layer_norm,
            weights: ln_1,
            saved: saved.ln_1,
            grads: ln_1_grads.reborrow(),
            d_normalized: &*d_hidden,
            direct: d_residual_after_attention,
            d_residual: d_residual_in,
            chunk_amax: d_residual_in_chunk_amax,
        })?;
        Ok(Some(chunk_count))
    }
}
