use super::args::Gpt2ForwardArgs;
use super::final_logits::{FinalForwardArgs, finish_forward};
use super::weights::Gpt2Weights;
use crate::types::{BlockForwardArgs, HiddenStateDevice};
use crate::uses_full_attention;
use cuda_core::DriverError;

pub(super) fn forward<'a>(
    weights: &Gpt2Weights,
    args: Gpt2ForwardArgs<'a>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    forward_with_hook(weights, args, &mut |_, _| Ok(()))
}

pub(super) fn forward_with_hook<'a>(
    weights: &Gpt2Weights,
    args: Gpt2ForwardArgs<'a>,
    hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    forward_impl(weights, args, hook, true)
}

pub(super) fn forward_impl<'a>(
    weights: &Gpt2Weights,
    args: Gpt2ForwardArgs<'a>,
    hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
    readout: bool,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    forward_with_boundaries(weights, args, hook, readout, &mut |_, _, _| Ok(false))
}

pub(super) fn forward_with_boundaries<'a>(
    weights: &Gpt2Weights,
    args: Gpt2ForwardArgs<'a>,
    hook: &mut dyn FnMut(usize, &mut HiddenStateDevice<'_>) -> Result<(), DriverError>,
    readout: bool,
    boundary: &mut dyn FnMut(usize, bool, &mut HiddenStateDevice<'_>) -> Result<bool, DriverError>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let Gpt2ForwardArgs {
        value_rms,
        value_projection_observer,
        mlp_paired,
        mlp_projection_observer,
        embeddings,
        attention_module,
        canon_module,
        attention_tc_module,
        quant_module,
        layer_norm_module,
        mlp_module,
        lm_head_module,
        tma_module,
        tma_pad,
        tma_descriptors,
        tma_weight_bytes_padded,
        mut hidden_nvfp4,
        mut attention_tc_scratch,
        mut mlp_activation_nvfp4,
        attention,
        block_ln_1,
        block_canon_a,
        block_ln_2,
        block_canon_c,
        mlp,
        ln_f,
        attention_qkv,
        attention_value_residual,
        attention_log_sum_exp,
        mlp_activation,
        mlp_route_scores,
        mlp_route_masks,
        logits,
        mut tape,
    } = args;

    let lm_head_weight_device = embeddings.token_embedding;

    let mut hidden = weights.embeddings.forward(embeddings)?;

    for (block_index, block) in weights.h.iter().enumerate() {
        hook(block_index, &mut hidden)?;
        hidden = block.forward_with_boundaries(
            BlockForwardArgs {
                value_rms,
                value_projection_observer,
                mlp_paired: &mut *mlp_paired,
                mlp_projection_observer,
                block_index,
                use_full_attention: uses_full_attention(block_index),
                attention_module,
                canon_module,
                attention_tc_module,
                quant_module,
                layer_norm_module,
                mlp_module,
                tma_module,
                tma_pad,
                hidden_nvfp4: hidden_nvfp4.reborrow(),
                attention_tc_scratch: attention_tc_scratch.reborrow(),
                mlp_activation_nvfp4: mlp_activation_nvfp4.reborrow(),
                tma_descriptors: &mut *tma_descriptors,
                tma_weight_bytes_padded: &mut *tma_weight_bytes_padded,
                projections: attention[block_index],
                ln_1: block_ln_1[block_index],
                canon_a: block_canon_a[block_index],
                ln_2: block_ln_2[block_index],
                canon_c: block_canon_c[block_index],
                mlp: mlp[block_index],
                qkv: &mut *attention_qkv,
                value_residual: &mut *attention_value_residual,
                attention_log_sum_exp: &mut *attention_log_sum_exp,
                mlp_activation: &mut *mlp_activation,
                mlp_route_scores: &mut *mlp_route_scores,
                mlp_route_masks: &mut *mlp_route_masks,
                hidden,
                tape: tape.as_mut().map(|tape| tape.block(block_index)),
            },
            boundary,
        )?;
    }

    boundary(weights.h.len() * 2, true, &mut hidden)?;
    hook(weights.h.len(), &mut hidden)?;
    if !readout {
        return Ok(hidden);
    }
    finish_forward(FinalForwardArgs {
        ln_f_weights: &weights.ln_f,
        layer_norm_module,
        quant_module,
        lm_head_module,
        lm_head_tma_module: tma_module,
        lm_head_tma_descriptors: tma_descriptors,
        ln_f,
        hidden_nvfp4,
        lm_head_weight_device,
        logits,
        tape,
        hidden,
    })
}
