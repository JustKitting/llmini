use cuda_core::{CudaStream, DeviceBuffer};

use crate::AppResult;
use crate::upload::{UploadedBlock, UploadedLinear, UploadedModel};

use super::record::UpdateSnapshotCollector;
use super::snapshot::PendingTensorUpdateDiagnostics;
use crate::training::grad_block::BlockGradBuffers;
use crate::training::grads::BackwardBuffers;
use crate::training::optimizer_state::{BlockState, LinearState, OptimizerStateBuffers};

pub(in crate::training::diagnostics) fn collect_update_snapshots(
    stream: &CudaStream,
    uploaded: &UploadedModel,
    grads: &BackwardBuffers,
    state: &OptimizerStateBuffers,
    step: u32,
    average_coefficient: f32,
    grad_scale: f32,
) -> AppResult<Vec<PendingTensorUpdateDiagnostics>> {
    let mut collector = UpdateSnapshotCollector::new(stream, step, average_coefficient, grad_scale);
    if let Some(adam) = state.token_embedding.as_adam() {
        collector.push_adam(
            "token_embedding",
            &uploaded.token_embedding,
            &grads.d_lm_head_weight,
            adam,
        )?;
    } else {
        collector.push_observed(
            "token_embedding",
            &uploaded.token_embedding,
            &grads.d_lm_head_weight,
        )?;
    }
    if gpt2_nvfp4::exclusive_self_attention_enabled() {
        collector.push_adam_with_weight_decay(
            "xsa_alphas",
            &uploaded.xsa_alphas,
            &grads.d_xsa_alphas,
            &state.xsa_alphas,
            0.0,
        )?;
    }
    collector.push_layer_norm("ln_f", &uploaded.ln_f, &grads.final_norm, &state.ln_f)?;

    for (index, ((block, grad), state)) in uploaded
        .blocks
        .iter()
        .zip(grads.blocks.iter())
        .zip(state.blocks.iter())
        .enumerate()
    {
        collect_block_updates(&mut collector, index, block, grad, state)?;
    }

    Ok(collector.into_updates())
}

fn collect_block_updates(
    collector: &mut UpdateSnapshotCollector<'_>,
    index: usize,
    block: &UploadedBlock,
    grad: &BlockGradBuffers,
    state: &BlockState,
) -> AppResult {
    macro_rules! collect_linear {
        ($name:literal, $linear:ident, $weight_grad:ident, $bias_grad:ident) => {
            collect_linear_updates(
                collector,
                &format!("block{index}.{}", $name),
                &block.$linear,
                &grad.$weight_grad,
                &grad.$bias_grad,
                &state.$linear,
            )?;
        };
    }

    collector.push_layer_norm(
        &format!("block{index}.ln_1"),
        &block.ln_1,
        &grad.ln_1,
        &state.ln_1,
    )?;
    collect_linear!("attn_qkv", attn_qkv, d_attn_qkv_weight, d_attn_qkv_bias);
    if gpt2_nvfp4::uses_full_attention(index) {
        collector.push_adam(
            &format!("block{index}.attn_qk_scale"),
            &block.attn_qk_scale,
            &grad.d_attn_qk_scale,
            &state.attn_qk_scale,
        )?;
    }
    collect_linear!(
        "attn_c_proj",
        attn_c_proj,
        d_attn_c_proj_weight,
        d_attn_c_proj_bias
    );
    collector.push_layer_norm(
        &format!("block{index}.ln_2"),
        &block.ln_2,
        &grad.ln_2,
        &state.ln_2,
    )?;
    collect_linear!("mlp_up", mlp_up, d_mlp_c_fc_weight, d_mlp_c_fc_bias);
    collect_linear!("mlp_down", mlp_down, d_mlp_c_proj_weight, d_mlp_c_proj_bias);
    Ok(())
}

fn collect_linear_updates(
    collector: &mut UpdateSnapshotCollector<'_>,
    name: &str,
    linear: &UploadedLinear,
    weight_grad: &DeviceBuffer<f32>,
    bias_grad: &DeviceBuffer<f32>,
    state: &LinearState,
) -> AppResult {
    collector.push_observed(&format!("{name}.weight"), &linear.weight, weight_grad)?;
    collector.push_adam(
        &format!("{name}.bias"),
        &linear.bias,
        bias_grad,
        &state.bias,
    )
}
