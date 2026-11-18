use cuda_core::DriverError;
use gpt2_nvfp4::{
    GPT2_MLP, GPT2_MLP_UP, GPT2_N_EMBD, GPT2_QKV, NEXTLAT_HIDDEN, NEXTLAT_INPUT,
    attention_trainable_qkv_dim, uses_full_attention,
};

use super::super::tensor::{AdamState, Fp32AdamState, MuonState, StateInit};
use crate::upload::{UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedNextLat};

pub(in crate::training) struct BlockState {
    pub(in crate::training) ln_1: LayerNormState,
    pub(in crate::training) canon_a: Fp32AdamState,
    pub(in crate::training) attn_qkv: LinearState,
    pub(in crate::training) attn_qk_scale: AdamState,
    pub(in crate::training) attn_c_proj: LinearState,
    pub(in crate::training) ln_2: LayerNormState,
    pub(in crate::training) canon_c: Fp32AdamState,
    pub(in crate::training) mlp_up: LinearState,
    pub(in crate::training) mlp_down: LinearState,
}

impl BlockState {
    pub(super) fn new(
        init: StateInit<'_>,
        block: &UploadedBlock,
        index: usize,
    ) -> Result<Self, DriverError> {
        let attn_qkv = LinearState::new(init, &block.attn_qkv, GPT2_QKV)?;
        attn_qkv.weight_muon.zero_inactive_tail(
            init,
            GPT2_N_EMBD * attention_trainable_qkv_dim(uses_full_attention(index)),
        )?;
        Ok(Self {
            ln_1: LayerNormState::new(init, &block.ln_1)?,
            canon_a: Fp32AdamState::new(init, &block.canon_a)?,
            attn_qkv,
            attn_qk_scale: AdamState::new(init, &block.attn_qk_scale)?,
            attn_c_proj: LinearState::new(init, &block.attn_c_proj, GPT2_N_EMBD)?,
            ln_2: LayerNormState::new(init, &block.ln_2)?,
            canon_c: Fp32AdamState::new(init, &block.canon_c)?,
            mlp_up: LinearState::new(init, &block.mlp_up, GPT2_MLP_UP)?,
            mlp_down: LinearState::new(init, &block.mlp_down, GPT2_MLP)?,
        })
    }

    pub(super) fn push_schedule_amax_ptrs(&self, pointers: &mut Vec<u64>) {
        self.ln_1.push_schedule_amax_ptrs(pointers);
        self.attn_qkv.push_schedule_amax_ptrs(pointers);
        pointers.push(self.attn_qk_scale.schedule_amax_ptr());
        self.attn_c_proj.push_schedule_amax_ptrs(pointers);
        self.ln_2.push_schedule_amax_ptrs(pointers);
        self.mlp_up.push_schedule_amax_ptrs(pointers);
        self.mlp_down.push_schedule_amax_ptrs(pointers);
    }
}

pub(in crate::training) struct NextLatState {
    pub(in crate::training) norm: LayerNormState,
    pub(in crate::training) input_projection: LinearState,
    pub(in crate::training) transition: LinearState,
    pub(in crate::training) output_projection: LinearState,
}

impl NextLatState {
    pub(super) fn new(
        init: StateInit<'_>,
        next_latent: &UploadedNextLat,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            norm: LayerNormState::new(init, &next_latent.norm)?,
            input_projection: LinearState::new(
                init,
                &next_latent.input_projection,
                NEXTLAT_INPUT.max(NEXTLAT_HIDDEN),
            )?,
            transition: LinearState::new(init, &next_latent.transition, NEXTLAT_HIDDEN)?,
            output_projection: LinearState::new(
                init,
                &next_latent.output_projection,
                NEXTLAT_HIDDEN.max(GPT2_N_EMBD),
            )?,
        })
    }

    pub(super) fn push_schedule_amax_ptrs(&self, pointers: &mut Vec<u64>) {
        self.norm.push_schedule_amax_ptrs(pointers);
        self.input_projection.push_schedule_amax_ptrs(pointers);
        self.transition.push_schedule_amax_ptrs(pointers);
        self.output_projection.push_schedule_amax_ptrs(pointers);
    }
}

pub(in crate::training) struct LayerNormState {
    pub(in crate::training) weight: AdamState,
    pub(in crate::training) bias: AdamState,
}

impl LayerNormState {
    pub(super) fn new(
        init: StateInit<'_>,
        layer_norm: &UploadedLayerNorm,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight: AdamState::new(init, &layer_norm.weight)?,
            bias: AdamState::new(init, &layer_norm.bias)?,
        })
    }

    pub(super) fn push_schedule_amax_ptrs(&self, pointers: &mut Vec<u64>) {
        pointers.push(self.weight.schedule_amax_ptr());
        pointers.push(self.bias.schedule_amax_ptr());
    }
}

pub(in crate::training) struct LinearState {
    pub(in crate::training) weight_muon: MuonState,
    pub(in crate::training) bias: AdamState,
}

impl LinearState {
    pub(super) fn new(
        init: StateInit<'_>,
        linear: &UploadedLinear,
        normuon_neurons: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            weight_muon: MuonState::new(init, &linear.weight, normuon_neurons)?,
            bias: AdamState::new(init, &linear.bias)?,
        })
    }

    pub(super) fn push_schedule_amax_ptrs(&self, pointers: &mut Vec<u64>) {
        pointers.push(self.bias.schedule_amax_ptr());
    }
}
