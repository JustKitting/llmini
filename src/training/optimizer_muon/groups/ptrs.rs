use cuda_core::DeviceBuffer;

use crate::upload::{UploadedModel, UploadedNvfp4};

use super::super::super::grads::BackwardBuffers;
use super::super::super::next_latent::NextLatGradBuffers;
use super::super::super::optimizer_state::{MuonState, OptimizerStateBuffers};
use super::HostPtrs;

macro_rules! block_linear_ptrs {
    ($name:ident, $linear:ident, $grad:ident) => {
        pub(super) fn $name(
            uploaded: &UploadedModel,
            grads: &BackwardBuffers,
            state: &OptimizerStateBuffers,
            i: usize,
        ) -> HostPtrs {
            linear(
                &uploaded.blocks[i].$linear.weight,
                &grads.blocks[i].$grad,
                &state.blocks[i].$linear.weight_muon,
            )
        }
    };
}

macro_rules! next_latent_linear_ptrs {
    ($name:ident, $linear:ident, $grad:ident) => {
        pub(super) fn $name(
            uploaded: &UploadedModel,
            grads: &NextLatGradBuffers,
            state: &OptimizerStateBuffers,
        ) -> HostPtrs {
            linear(
                &uploaded.next_latent.$linear.weight,
                &grads.$grad,
                &state.next_latent.$linear.weight_muon,
            )
        }
    };
}

block_linear_ptrs!(qkv, attn_qkv, d_attn_qkv_weight);
block_linear_ptrs!(c_proj, attn_c_proj, d_attn_c_proj_weight);
block_linear_ptrs!(mlp_up, mlp_up, d_mlp_c_fc_weight);
block_linear_ptrs!(mlp_down, mlp_down, d_mlp_c_proj_weight);

next_latent_linear_ptrs!(
    next_latent_input_projection,
    input_projection,
    d_input_projection_weight
);
next_latent_linear_ptrs!(next_latent_transition, transition, d_transition_weight);
next_latent_linear_ptrs!(
    next_latent_output_projection,
    output_projection,
    d_output_projection_weight
);

fn linear(weight: &UploadedNvfp4, grad: &DeviceBuffer<f32>, state: &MuonState) -> HostPtrs {
    HostPtrs {
        grad: grad.cu_deviceptr(),
        momentum: state.momentum.cu_deviceptr(),
        variance: state.variance.cu_deviceptr(),
        second_momentum: state.second_momentum.cu_deviceptr(),
        z_master: state.z_master.cu_deviceptr(),
        x_master: state.x_master.cu_deviceptr(),
        schedule_amax: state.schedule_amax.cu_deviceptr(),
        bytes: weight.bytes.cu_deviceptr(),
        scales: weight.scales.cu_deviceptr(),
        global_scale: weight.global_scale.cu_deviceptr(),
        rows: 0,
        cols: 0,
        learning_rate_multiplier: 1.0,
        qk_clip_factor_offset: u32::MAX,
        qk_clip_head_dim: 0,
    }
}
