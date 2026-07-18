use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_QK_SCALE_STORAGE, GPT2_QKV};

use crate::training::grad_block::BlockGradBuffers;

use super::{HostGradView, push_layer_norm_views, push_prefixed_views};

pub(super) fn push_block_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    block_index: usize,
    block: &'a BlockGradBuffers,
) {
    let prefix = format!("blocks.{block_index}");
    push_layer_norm_views(rows, &format!("{prefix}.ln_1"), &block.ln_1);
    push_attention_views(rows, &prefix, block_index, block);
    push_layer_norm_views(rows, &format!("{prefix}.ln_2"), &block.ln_2);
    push_mlp_views(rows, &prefix, block);
}

fn push_attention_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    prefix: &str,
    block_index: usize,
    block: &'a BlockGradBuffers,
) {
    push_prefixed_views(
        rows,
        prefix,
        &[
            (
                "attn_qkv.weight",
                &block.d_attn_qkv_weight,
                GPT2_N_EMBD * GPT2_QKV,
            ),
            ("attn_qkv.bias", &block.d_attn_qkv_bias, GPT2_QKV),
            (
                "attn_c_proj.weight",
                &block.d_attn_c_proj_weight,
                GPT2_N_EMBD * GPT2_N_EMBD,
            ),
            ("attn_c_proj.bias", &block.d_attn_c_proj_bias, GPT2_N_EMBD),
        ],
    );
    if gpt2_nvfp4::uses_full_attention(block_index) {
        push_prefixed_views(
            rows,
            prefix,
            &[(
                "attn_qk_scale",
                &block.d_attn_qk_scale,
                GPT2_QK_SCALE_STORAGE,
            )],
        );
    }
}

fn push_mlp_views<'a>(rows: &mut Vec<HostGradView<'a>>, prefix: &str, block: &'a BlockGradBuffers) {
    push_prefixed_views(
        rows,
        prefix,
        &[
            (
                "mlp_up.weight",
                &block.d_mlp_c_fc_weight,
                GPT2_N_EMBD * GPT2_MLP,
            ),
            ("mlp_up.bias", &block.d_mlp_c_fc_bias, GPT2_MLP),
            (
                "mlp_down.weight",
                &block.d_mlp_c_proj_weight,
                GPT2_MLP * GPT2_N_EMBD,
            ),
            ("mlp_down.bias", &block.d_mlp_c_proj_bias, GPT2_N_EMBD),
        ],
    );
}
