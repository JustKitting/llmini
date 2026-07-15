use cuda_core::DriverError;

use super::super::tensor::{AdamState, MuonState, StateInit};
use crate::upload::{UploadedBlock, UploadedLayerNorm, UploadedLinear, UploadedNextLat};

pub(in crate::training) struct BlockState {
    pub(in crate::training) ln_1: LayerNormState,
    pub(in crate::training) attn_qkv: LinearState,
    pub(in crate::training) attn_c_proj: LinearState,
    pub(in crate::training) ln_2: LayerNormState,
    pub(in crate::training) mlp_up: LinearState,
    pub(in crate::training) mlp_down: LinearState,
}

impl BlockState {
    pub(super) fn new(init: StateInit<'_>, block: &UploadedBlock) -> Result<Self, DriverError> {
        Ok(Self {
            ln_1: LayerNormState::new(init, &block.ln_1)?,
            attn_qkv: LinearState::new(init, &block.attn_qkv)?,
            attn_c_proj: LinearState::new(init, &block.attn_c_proj)?,
            ln_2: LayerNormState::new(init, &block.ln_2)?,
            mlp_up: LinearState::new(init, &block.mlp_up)?,
            mlp_down: LinearState::new(init, &block.mlp_down)?,
        })
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
            input_projection: LinearState::new(init, &next_latent.input_projection)?,
            transition: LinearState::new(init, &next_latent.transition)?,
            output_projection: LinearState::new(init, &next_latent.output_projection)?,
        })
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
}

pub(in crate::training) struct LinearState {
    pub(in crate::training) weight_muon: MuonState,
    pub(in crate::training) bias: AdamState,
}

impl LinearState {
    pub(super) fn new(init: StateInit<'_>, linear: &UploadedLinear) -> Result<Self, DriverError> {
        Ok(Self {
            weight_muon: MuonState::new(init, &linear.weight)?,
            bias: AdamState::new(init, &linear.bias)?,
        })
    }
}
