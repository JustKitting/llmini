use std::cmp::Reverse;

use gpt2_nvfp4::{
    AttentionDims, GPT2_MLP, GPT2_MLP_UP, GPT2_N_EMBD, NEXTLAT_HIDDEN, NEXTLAT_INPUT,
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
    engram: Option<&crate::training::engram::Engram>,
    feedback: Option<&crate::training::latent_feedback::Feedback>,
    mhc: Option<&crate::training::mhc::Mhc>,
) -> Vec<HostPtrs> {
    let mut rows = all_slots(uploaded, grads, next_latent_grads, state);
    if let (Some(e), Some(w)) = (engram, uploaded.engram.as_ref()) {
        for (e, w) in e.modules.iter().zip(&w.modules).filter(|(_, w)| w.cfg.layer < state.active_depth) {
            for (slot, tensor) in w.cfg.projector_indices().into_iter().enumerate() {
                rows.push(
                    ptrs::linear(&w.nv[tensor], &e.ng[tensor], &e.projector_states[slot])
                        .shape(w.cfg.memory(), GPT2_N_EMBD),
                );
            }
        }
    }
    if let (Some(f), Some(w)) = (feedback, uploaded.latent_feedback.as_ref()) {
        rows.push(ptrs::linear(&w.weight, &f.grad, &f.state).shape(GPT2_N_EMBD, GPT2_N_EMBD));
    }
    if let (Some(m), Some(w)) = (mhc, uploaded.mhc.as_ref()) {
        for index in m.active_unit_indices() {
            let u = &m.units[index];
            let w = &w.units[index];
            rows.push(
                ptrs::linear(&w.phi, &u.grad, &u.phi)
                    .shape(crate::training::mhc::WIDTH, crate::training::mhc::PAD),
            );
        }
    }
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
    let depth = state.active_depth;
    for i in 0..depth {
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
    append(&mut rows, depth, GPT2_N_EMBD, GPT2_N_EMBD, |i| {
        ptrs::c_proj(uploaded, grads, state, i)
    });
    append(&mut rows, depth, GPT2_N_EMBD, GPT2_MLP_UP, |i| {
        ptrs::mlp_up(uploaded, grads, state, i)
    });
    append(&mut rows, depth, GPT2_MLP, GPT2_N_EMBD, |i| {
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

fn append<F>(rows: &mut Vec<HostPtrs>, depth: usize, input_dim: usize, output_dim: usize, ptrs: F)
where
    F: Fn(usize) -> HostPtrs,
{
    for i in 0..depth {
        rows.push(ptrs(i).shape(input_dim, output_dim));
    }
}
