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
    let Gpt2ForwardArgs {
        embeddings,
        attention_module,
        attention_tc_module,
        quant_module,
        layer_norm_module,
        mlp_module,
        lm_head_module,
        tma_module,
        tma_scale_pack,
        tma_pad,
        tma_descriptors,
        tma_input_scale_packed,
        tma_wide_input_scale_packed,
        tma_weight_scale_packed,
        tma_weight_bytes_padded,
        mut hidden_nvfp4,
        mut attention_tc_scratch,
        mut mlp_activation_nvfp4,
        attention,
        block_ln_1,
        block_ln_2,
        mlp,
        ln_f,
        attention_qkv,
        attention_value_residual,
        attention_log_sum_exp,
        mlp_activation,
        logits,
        mut tape,
    } = args;

    let lm_head_weight_device = embeddings.token_embedding;

    let mut hidden = weights.embeddings.forward(embeddings)?;

    for (block_index, block) in weights.h.iter().enumerate() {
        hidden = block.forward(BlockForwardArgs {
            block_index,
            use_full_attention: uses_full_attention(block_index),
            attention_module,
            attention_tc_module,
            quant_module,
            layer_norm_module,
            mlp_module,
            tma_module,
            tma_scale_pack,
            tma_pad,
            hidden_nvfp4: hidden_nvfp4.reborrow(),
            attention_tc_scratch: attention_tc_scratch.reborrow(),
            mlp_activation_nvfp4: mlp_activation_nvfp4.reborrow(),
            tma_descriptors: &mut *tma_descriptors,
            tma_input_scale_packed: &mut *tma_input_scale_packed,
            tma_wide_input_scale_packed: &mut *tma_wide_input_scale_packed,
            tma_weight_scale_packed: &mut *tma_weight_scale_packed,
            tma_weight_bytes_padded: &mut *tma_weight_bytes_padded,
            projections: attention[block_index],
            ln_1: block_ln_1[block_index],
            ln_2: block_ln_2[block_index],
            mlp: mlp[block_index],
            qkv: &mut *attention_qkv,
            value_residual: &mut *attention_value_residual,
            attention_log_sum_exp: &mut *attention_log_sum_exp,
            mlp_activation: &mut *mlp_activation,
            hidden,
            tape: tape.as_mut().map(|tape| tape.block(block_index)),
        })?;
    }

    finish_forward(FinalForwardArgs {
        ln_f_weights: &weights.ln_f,
        layer_norm_module,
        quant_module,
        lm_head_module,
        lm_head_tma_module: tma_module,
        lm_head_tma_scale_pack: tma_scale_pack,
        lm_head_tma_descriptors: tma_descriptors,
        lm_head_input_scale_packed: tma_input_scale_packed,
        lm_head_weight_scale_packed: tma_weight_scale_packed,
        ln_f,
        hidden_nvfp4,
        lm_head_weight_device,
        logits,
        tape,
        hidden,
    })
}
