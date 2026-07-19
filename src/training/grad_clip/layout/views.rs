use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    GPT2_N_EMBD, GPT2_VOCAB_SIZE, GPT2_XSA_ALPHA_STORAGE, exclusive_self_attention_enabled,
};

use crate::training::{
    grad_block::LayerNormGradBuffers, grads::BackwardBuffers, next_latent::NextLatGradBuffers,
};

mod blocks;
mod next_latent;

use blocks::push_block_views;
use next_latent::push_next_latent_views;

pub(super) struct HostGradView<'a> {
    pub(super) name: String,
    pub(super) buffer: &'a DeviceBuffer<f32>,
    pub(super) len: usize,
}

pub(super) fn parameter_gradient_views<'a>(
    grads: &'a BackwardBuffers,
    next_latent: &'a NextLatGradBuffers,
) -> Vec<HostGradView<'a>> {
    let mut rows = Vec::new();
    push_view(
        &mut rows,
        "lm_head.weight",
        &grads.d_lm_head_weight,
        GPT2_VOCAB_SIZE * GPT2_N_EMBD,
    );
    if exclusive_self_attention_enabled() {
        push_view(
            &mut rows,
            "xsa_alphas",
            &grads.d_xsa_alphas,
            GPT2_XSA_ALPHA_STORAGE,
        );
    }
    push_layer_norm_views(&mut rows, "final_norm", &grads.final_norm);

    for (block_index, block) in grads.blocks.iter().enumerate() {
        push_block_views(&mut rows, block_index, block);
    }
    push_next_latent_views(&mut rows, next_latent);

    rows
}

fn push_layer_norm_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    name: &str,
    grads: &'a LayerNormGradBuffers,
) {
    push_prefixed_views(
        rows,
        name,
        &[
            ("weight", &grads.d_weight, GPT2_N_EMBD),
            ("bias", &grads.d_bias, GPT2_N_EMBD),
        ],
    );
}

fn push_prefixed_views<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    prefix: &str,
    views: &[(&str, &'a DeviceBuffer<f32>, usize)],
) {
    for &(name, buffer, len) in views {
        push_view(rows, &format!("{prefix}.{name}"), buffer, len);
    }
}

fn push_view<'a>(
    rows: &mut Vec<HostGradView<'a>>,
    name: &str,
    buffer: &'a DeviceBuffer<f32>,
    len: usize,
) {
    rows.push(HostGradView::new(name, buffer, len));
}

impl<'a> HostGradView<'a> {
    fn new(name: &str, buffer: &'a DeviceBuffer<f32>, len: usize) -> Self {
        Self {
            name: name.to_string(),
            buffer,
            len,
        }
    }
}
