use std::cmp::Reverse;

use gpt2_nvfp4::{
    AttentionDims, GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, NEXTLAT_HIDDEN, NEXTLAT_INPUT,
    attention_trainable_qkv_dim, uses_full_attention,
};
use rust_kernels_cuda::optimizer::MUON_MATRIX_PHASES;

use super::super::MUON_MATRIX_SLOTS;
use super::{HostPtrs, padding::MuonPaddingBuffers, ptrs};
use crate::{
    training::{
        grads::BackwardBuffers, learning_rate, next_latent::NextLatGradBuffers,
        optimizer_state::OptimizerStateBuffers,
    },
    upload::UploadedModel,
};

pub(super) fn build_slots(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
    padding: &MuonPaddingBuffers,
) -> Vec<HostPtrs> {
    let mut rows = all_slots(uploaded, grads, next_latent_grads, state);
    schedule_slots(&mut rows);
    pad_slots(&mut rows, padding);
    rows
}

fn all_slots(
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
    state: &OptimizerStateBuffers,
) -> Vec<HostPtrs> {
    let mut rows = Vec::with_capacity(MUON_MATRIX_SLOTS);
    for i in 0..GPT2_N_LAYER {
        let dims = AttentionDims::new(uses_full_attention(i));
        rows.push(
            ptrs::qkv(uploaded, grads, state, i)
                .shape(
                    GPT2_N_EMBD,
                    attention_trainable_qkv_dim(uses_full_attention(i)),
                )
                .qk_clip(i * dims.head_count as usize, dims.head_dim as usize),
        );
    }
    append(&mut rows, GPT2_N_EMBD, GPT2_N_EMBD, |i| {
        ptrs::c_proj(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_N_EMBD, GPT2_MLP, |i| {
        ptrs::mlp_up(uploaded, grads, state, i)
    });
    append(&mut rows, GPT2_MLP, GPT2_N_EMBD, |i| {
        ptrs::mlp_down(uploaded, grads, state, i)
    });
    rows.push(
        ptrs::next_latent_input_projection(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(learning_rate::next_latent_scale())
            .shape(NEXTLAT_INPUT, NEXTLAT_HIDDEN),
    );
    rows.push(
        ptrs::next_latent_transition(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(learning_rate::next_latent_scale())
            .shape(NEXTLAT_HIDDEN, NEXTLAT_HIDDEN),
    );
    rows.push(
        ptrs::next_latent_output_projection(uploaded, next_latent_grads, state)
            .learning_rate_multiplier(learning_rate::next_latent_scale())
            .shape(NEXTLAT_HIDDEN, GPT2_N_EMBD),
    );
    rows
}

fn schedule_slots(rows: &mut [HostPtrs]) {
    rows.sort_by_key(|slot| Reverse(slot.estimated_polar_work()));
}

fn pad_slots(rows: &mut Vec<HostPtrs>, padding: &MuonPaddingBuffers) {
    while !rows.len().is_multiple_of(MUON_MATRIX_PHASES) {
        rows.push(padding.ptrs());
    }
}

fn append<F>(rows: &mut Vec<HostPtrs>, row_count: usize, col_count: usize, ptrs: F)
where
    F: Fn(usize) -> HostPtrs,
{
    for i in 0..GPT2_N_LAYER {
        rows.push(ptrs(i).shape(row_count, col_count));
    }
}
