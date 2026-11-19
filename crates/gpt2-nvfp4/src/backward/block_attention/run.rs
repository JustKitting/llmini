use cuda_core::DriverError;
use rust_kernels_cuda::attention::{
    AccumulateValueResidualGradArgs, AttentionOutputGate, ExclusiveSelfAttentionBackwardArgs,
    FinishValueResidualGradArgs, HeadwiseAttentionGateBackwardArgs,
    InitializeValueResidualGradArgs,
};
use rust_kernels_cuda::canon::CanonBackwardArgs;

use super::types::BlockAttentionBackwardArgs;
use crate::backward::{
    AttentionCProjBackwardArgs, AttentionCoreBackwardArgs, AttentionQkvBackwardArgs,
    Gpt2LayerNormBackwardAddAmaxArgs, Gpt2LayerNormBackwardAddArgs, Gpt2LayerNormBackwardAmaxArgs,
    attention_c_proj_backward, causal_attention_backward, layer_norm_backward_add,
    layer_norm_backward_add_amax, layer_norm_backward_amax, qkv_projection_backward,
};
use crate::types::{BlockBackwardGrads, project_attention_value};
use crate::{
    AttentionDims, GPT2_N_LAYER, attention_headwise_gate_enabled, attention_headwise_gate_offset,
    uses_exclusive_self_attention, uses_value_residual,
};

pub fn attention_side_backward(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
) -> Result<Option<u32>, DriverError> {
    attention_side_backward_impl(args, true)
}
pub fn attention_side_backward_branch(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
) -> Result<Option<u32>, DriverError> {
    attention_side_backward_impl(args, false)
}
fn attention_side_backward_impl(
    args: BlockAttentionBackwardArgs<'_, '_, '_>,
    add_residual: bool,
) -> Result<Option<u32>, DriverError> {
    let BlockAttentionBackwardArgs {
        block_index,
        use_full_attention,
        reuse_forward_probs,
        stream,
        modules,
        saved,
        ln_1,
        canon_a,
        projections,
        d_residual_after_attention,
        precomputed_d_residual_after_attention_amax_chunks,
        d_residual_in,
        d_residual_in_chunk_amax,
        d_hidden,
        d_qkv,
        d_value_residual,
        d_xsa_alphas,
        grads,
        scratch,
        seeds,
    } = args;
    let BlockBackwardGrads {
        ln_1: mut ln_1_grads,
        d_canon_a_weight,
        d_attn_qkv_weight,
        d_attn_qkv_bias,
        d_attn_qk_scale,
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
    let gate_amax_offset = if use_full_attention {
        3 * saved.row_count
    } else {
        4 * saved.row_count
    };
    let use_xsa = uses_exclusive_self_attention(use_full_attention);
    let gate = crate::config::attention_output_gate(use_full_attention);
    let xsa_amax_chunks = if use_xsa {
        let raw_out = if use_full_attention {
            saved.attention_out
        } else {
            saved
                .headwise_gate_input
                .expect("KDA XSA requires a saved raw-output tape")
        };
        let dims = AttentionDims::new(use_full_attention);
        Some(modules.attention.exclusive_self_attention_backward(
            ExclusiveSelfAttentionBackwardArgs {
                stream,
                qkv_f16: saved.qkv,
                raw_out_f16: raw_out,
                d_out: &mut *d_hidden,
                d_qkv: &mut *d_qkv,
                d_qkv_chunk_amax: &mut *scratch.qkv.linear.e_h.chunk_amax,
                d_xsa_alphas,
                xsa_alphas: projections.xsa_alphas,
                row_count: saved.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
                head_count: dims.head_count,
                head_dim: dims.head_dim,
                value_offset: 2 * dims.embedding_dim,
                gate_offset: attention_headwise_gate_offset(use_full_attention) as u32,
                gate_amax_offset,
                alpha_offset: projections.xsa_alpha_offset,
                gate,
                kda_value_activation: !use_full_attention && !crate::GPT2_VALUE_RMS,
            },
        )?)
    } else if gate != AttentionOutputGate::Disabled {
        let raw_out = if use_full_attention {
            saved.attention_out
        } else {
            saved
                .headwise_gate_input
                .expect("KDA headwise gate requires a saved raw-output tape")
        };
        let dims = AttentionDims::new(use_full_attention);
        modules
            .attention
            .headwise_attention_gate_backward(HeadwiseAttentionGateBackwardArgs {
                stream,
                qkv_f16: saved.qkv,
                raw_out_f16: raw_out,
                d_out: &mut *d_hidden,
                d_qkv: &mut *d_qkv,
                d_qkv_chunk_amax: &mut *scratch.qkv.linear.e_h.chunk_amax,
                gate_amax_offset,
                row_count: saved.row_count,
                embedding_dim: dims.embedding_dim,
                qkv_dim: dims.qkv_dim,
                head_count: dims.head_count,
                head_dim: dims.head_dim,
                gate_offset: attention_headwise_gate_offset(use_full_attention) as u32,
                gate,
            })?;
        None
    } else {
        None
    };
    let d_qkv_amax_chunks = causal_attention_backward(AttentionCoreBackwardArgs {
        block_index,
        use_full_attention,
        reuse_forward_probs,
        stream,
        module: modules.attention,
        tc_module: modules.f16_tc,
        saved,
        d_attention_out: &*d_hidden,
        qk_scale: projections.qk_scale,
        d_qkv,
        d_qkv_chunk_amax: &mut *scratch.qkv.linear.e_h.chunk_amax,
        d_qk_scale: d_attn_qk_scale,
        accumulate_value_grad: use_xsa,
        scratch: scratch.core,
        backward_mask_seed: seeds.qkv.attention_mask_seed(),
    })?;
    if use_xsa || gate != AttentionOutputGate::Disabled {
        assert_eq!(
            d_qkv_amax_chunks,
            Some(gate_amax_offset),
            "gate amax chunks must immediately follow the attention-core chunks"
        );
    }
    let dims = AttentionDims::new(use_full_attention);
    if crate::GPT2_VALUE_RMS {
        let raw = scratch.value_rms;
        // Only the raw V section is read after this replay. Keep the full QKV
        // row stride so sharing, normalization and their observers use the
        // same coordinates as forward.
        project_attention_value(
            stream,
            modules.tma,
            modules.pad,
            raw.descriptors,
            raw.padded_weight,
            saved.qkv_input_nvfp4,
            projections,
            raw.qkv,
            saved.row_count,
            dims,
        )?;
        if uses_value_residual(block_index) {
            modules.attention.mix_value_residual(
                rust_kernels_cuda::attention::MixValueResidualArgs {
                    stream,
                    qkv: raw.qkv,
                    first_value: raw.raw_first_value,
                    row_count: saved.row_count,
                    embedding_dim: dims.embedding_dim,
                    qkv_dim: dims.qkv_dim,
                },
            )?;
        }
        if let Some(observer) = raw.observer {
            observer.observe(
                block_index,
                stream,
                raw.qkv,
                saved.row_count as usize,
                dims.embedding_dim as usize,
                dims.qkv_dim as usize,
                true,
            )?;
            observer.represented_point(
                block_index,
                stream,
                raw.qkv,
                saved.qkv,
                saved.row_count as usize,
                dims.embedding_dim as usize,
                dims.qkv_dim as usize,
                !use_full_attention,
            )?;
        }
        // Core and XSA have both contributed in normalized-V coordinates.
        modules.value_rms.backward(
            stream,
            raw.qkv,
            d_qkv,
            saved.row_count,
            dims.embedding_dim,
            dims.qkv_dim,
            !use_full_attention,
        )?;
    }
    let routes_value_residual = uses_value_residual(block_index);
    if routes_value_residual && block_index == GPT2_N_LAYER - 1 {
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
    } else if routes_value_residual {
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
        // RMS and fixed sharing change dQKV after the core's amax measurement.
        precomputed_d_qkv_amax_chunks: if crate::GPT2_VALUE_RMS
            || block_index == 0
            || routes_value_residual
        {
            None
        } else if let Some(xsa_amax_chunks) = xsa_amax_chunks {
            Some(
                d_qkv_amax_chunks.expect("attention core must report gradient amax chunks")
                    + xsa_amax_chunks,
            )
        } else if attention_headwise_gate_enabled() {
            Some(
                d_qkv_amax_chunks.expect("attention core must report gradient amax chunks")
                    + saved.row_count,
            )
        } else {
            d_qkv_amax_chunks
        },
        scratch: scratch.qkv,
        seeds: seeds.qkv,
    })?;
    let d_ln_1_normalized = if crate::canon_ac_enabled() {
        modules.canon.backward(CanonBackwardArgs {
            stream,
            residual_f16: saved.ln_1.residual,
            mean: saved.ln_1.mean,
            inv_std: saved.ln_1.inv_std,
            norm_weight: ln_1.weight,
            norm_bias: ln_1.bias,
            canon_weight: canon_a.weight,
            d_output: &*d_hidden,
            d_input: &mut *d_qkv,
            d_weight: d_canon_a_weight,
            row_count: saved.row_count,
            seq_len: saved.seq_len,
            width: crate::GPT2_EMBEDDING_DIM,
            norm_output_scale: crate::layer_norm_scale(block_index),
        })?;
        &*d_qkv
    } else {
        &*d_hidden
    };
    if !add_residual {
        return Ok(Some(layer_norm_backward_amax(
            Gpt2LayerNormBackwardAmaxArgs {
                stream,
                module: modules.layer_norm,
                weights: ln_1,
                saved: saved.ln_1,
                grads: ln_1_grads.reborrow(),
                d_normalized: d_ln_1_normalized,
                d_residual: d_residual_in,
                chunk_amax: d_residual_in_chunk_amax,
                output_scale: crate::layer_norm_scale(block_index),
            },
        )?));
    }
    if block_index == 0 {
        layer_norm_backward_add(Gpt2LayerNormBackwardAddArgs {
            stream,
            module: modules.layer_norm,
            weights: ln_1,
            saved: saved.ln_1,
            grads: ln_1_grads.reborrow(),
            d_normalized: d_ln_1_normalized,
            direct: d_residual_after_attention,
            d_residual: d_residual_in,
            output_scale: crate::layer_norm_scale(block_index),
        })?;
        Ok(None)
    } else {
        let chunk_count = layer_norm_backward_add_amax(Gpt2LayerNormBackwardAddAmaxArgs {
            stream,
            module: modules.layer_norm,
            weights: ln_1,
            saved: saved.ln_1,
            grads: ln_1_grads.reborrow(),
            d_normalized: d_ln_1_normalized,
            direct: d_residual_after_attention,
            d_residual: d_residual_in,
            chunk_amax: d_residual_in_chunk_amax,
            output_scale: crate::layer_norm_scale(block_index),
        })?;
        Ok(Some(chunk_count))
    }
}
