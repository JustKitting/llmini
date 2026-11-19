use cuda_core::{CudaStream, DriverError};

use super::tensor::Materializer;
use crate::training::runtime::Runtime;
use crate::upload::{UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat};

use super::super::grad_block::{BlockGradBuffers, LayerNormGradBuffers};
use super::super::grads::BackwardBuffers;
use super::super::learning_rate::schedule_free_beta;
use super::super::next_latent::NextLatGradBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{
    BlockState, LayerNormState, LinearState, NextLatState, OptimizerStateBuffers,
};

pub(in crate::training) fn materialize_training_weights(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    scratch: &mut OptimizerScratch,
    state: &OptimizerStateBuffers,
    grads: &BackwardBuffers,
    next_latent_grads: &NextLatGradBuffers,
) -> Result<(), DriverError> {
    materialize_training_graph(
        stream,
        runtime,
        uploaded,
        scratch,
        state,
        grads,
        Some(next_latent_grads),
    )
}

pub(in crate::training) fn materialize_backbone_training_weights(
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    scratch: &mut OptimizerScratch,
    state: &OptimizerStateBuffers,
    grads: &BackwardBuffers,
) -> Result<(), DriverError> {
    materialize_training_graph(
        &runtime.stream,
        runtime,
        uploaded,
        scratch,
        state,
        grads,
        None,
    )
}

fn materialize_training_graph(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    scratch: &mut OptimizerScratch,
    state: &OptimizerStateBuffers,
    grads: &BackwardBuffers,
    next_latent_grads: Option<&NextLatGradBuffers>,
) -> Result<(), DriverError> {
    if state.training_weights_ready() {
        return Ok(());
    }
    let beta = schedule_free_beta(state.next_step());
    let symexp_lin_beta = super::super::symexp_lin::beta();
    let reuse_adam_amax = state.next_step() > 1;
    let reuse_muon_amax = state.next_step() > 1
        && (symexp_lin_beta == 0.0 || super::super::symexp_lin::reuse_precomputed_amax());
    let mut materializer = Materializer::new(
        stream,
        &runtime.optimizer,
        &runtime.f32_ops,
        scratch,
        beta,
        symexp_lin_beta,
        reuse_adam_amax,
        state.scheduled_values_valid(),
    );

    if next_latent_grads.is_some() {
        materializer.token_embedding(
            &mut uploaded.token_embedding,
            &state.token_embedding,
            &grads.d_lm_head_weight,
        )?;
    }
    if gpt2_nvfp4::exclusive_self_attention_enabled() {
        materializer.adam(
            &mut uploaded.xsa_alphas,
            &state.xsa_alphas,
            &grads.d_xsa_alphas,
        )?;
    }
    if let Some(next_latent_grads) = next_latent_grads {
        materialize_layer_norm(
            &mut materializer,
            &mut uploaded.ln_f,
            &state.ln_f,
            &grads.final_norm,
        )?;
        materialize_next_latent(
            &mut materializer,
            &mut uploaded.next_latent,
            &state.next_latent,
            next_latent_grads,
            reuse_muon_amax,
        )?;
    }

    for (block_index, ((block, state), grads)) in uploaded
        .blocks
        .iter_mut()
        .zip(state.blocks.iter())
        .zip(grads.blocks.iter())
        .enumerate()
    {
        materialize_block(
            &mut materializer,
            block_index,
            block,
            state,
            grads,
            reuse_muon_amax,
        )?;
    }

    state.mark_training_weights_ready();
    Ok(())
}

/// Quantize the averaged schedule-free state used for held-out evaluation.
/// This must not use the next-step `z`/`x` training interpolation.
pub(in crate::training) fn materialize_evaluation_weights(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    scratch: &mut OptimizerScratch,
    state: &OptimizerStateBuffers,
) -> Result<(), DriverError> {
    state.invalidate_training_weights();
    let mut materializer = Materializer::new(
        stream,
        &runtime.optimizer,
        &runtime.f32_ops,
        scratch,
        0.0,
        super::super::symexp_lin::beta(),
        false,
        false,
    );

    materializer.master(
        &mut uploaded.token_embedding,
        state.token_embedding.x_master(),
    )?;
    if gpt2_nvfp4::exclusive_self_attention_enabled() {
        materializer.master(&mut uploaded.xsa_alphas, &state.xsa_alphas.x_master)?;
    }
    materialize_evaluation_layer_norm(&mut materializer, &mut uploaded.ln_f, &state.ln_f)?;
    materialize_evaluation_next_latent(
        &mut materializer,
        &mut uploaded.next_latent,
        &state.next_latent,
    )?;

    for (block_index, (block, state)) in uploaded
        .blocks
        .iter_mut()
        .zip(state.blocks.iter())
        .enumerate()
    {
        materialize_evaluation_block(&mut materializer, block_index, block, state)?;
    }

    Ok(())
}

fn materialize_next_latent(
    materializer: &mut Materializer<'_>,
    next_latent: &mut UploadedNextLat,
    state: &NextLatState,
    grads: &NextLatGradBuffers,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    materialize_layer_norm_from_parts(
        materializer,
        &mut next_latent.norm,
        &state.norm,
        &grads.d_norm_weight,
        &grads.d_norm_bias,
    )?;
    materialize_linear(
        materializer,
        &mut next_latent.input_projection,
        &state.input_projection,
        &grads.d_input_projection_weight,
        &grads.d_input_projection_bias,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut next_latent.transition,
        &state.transition,
        &grads.d_transition_weight,
        &grads.d_transition_bias,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut next_latent.output_projection,
        &state.output_projection,
        &grads.d_output_projection_weight,
        &grads.d_output_projection_bias,
        reuse_muon_amax,
    )
}

fn materialize_block(
    materializer: &mut Materializer<'_>,
    block_index: usize,
    block: &mut crate::upload::UploadedBlock,
    state: &BlockState,
    grads: &BlockGradBuffers,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    materialize_layer_norm(materializer, &mut block.ln_1, &state.ln_1, &grads.ln_1)?;
    if gpt2_nvfp4::canon_ac_enabled() {
        materializer.fp32_adam(&mut block.canon_a, &state.canon_a)?;
    }
    materialize_linear(
        materializer,
        &mut block.attn_qkv,
        &state.attn_qkv,
        &grads.d_attn_qkv_weight,
        &grads.d_attn_qkv_bias,
        reuse_muon_amax,
    )?;
    if gpt2_nvfp4::uses_full_attention(block_index) {
        materializer.adam(
            &mut block.attn_qk_scale,
            &state.attn_qk_scale,
            &grads.d_attn_qk_scale,
        )?;
    }
    materialize_linear(
        materializer,
        &mut block.attn_c_proj,
        &state.attn_c_proj,
        &grads.d_attn_c_proj_weight,
        &grads.d_attn_c_proj_bias,
        reuse_muon_amax,
    )?;
    materialize_layer_norm(materializer, &mut block.ln_2, &state.ln_2, &grads.ln_2)?;
    if gpt2_nvfp4::canon_ac_enabled() {
        materializer.fp32_adam(&mut block.canon_c, &state.canon_c)?;
    }
    materialize_linear(
        materializer,
        &mut block.mlp_up,
        &state.mlp_up,
        &grads.d_mlp_c_fc_weight,
        &grads.d_mlp_c_fc_bias,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut block.mlp_down,
        &state.mlp_down,
        &grads.d_mlp_c_proj_weight,
        &grads.d_mlp_c_proj_bias,
        reuse_muon_amax,
    )
}

fn materialize_layer_norm(
    materializer: &mut Materializer<'_>,
    layer_norm: &mut UploadedLayerNorm,
    state: &LayerNormState,
    grads: &LayerNormGradBuffers,
) -> Result<(), DriverError> {
    materialize_layer_norm_from_parts(
        materializer,
        layer_norm,
        state,
        &grads.d_weight,
        &grads.d_bias,
    )
}

fn materialize_layer_norm_from_parts(
    materializer: &mut Materializer<'_>,
    layer_norm: &mut UploadedLayerNorm,
    state: &LayerNormState,
    weight_grad: &cuda_core::DeviceBuffer<f32>,
    bias_grad: &cuda_core::DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    materializer.adam(&mut layer_norm.weight, &state.weight, weight_grad)?;
    materializer.adam(&mut layer_norm.bias, &state.bias, bias_grad)
}

fn materialize_linear(
    materializer: &mut Materializer<'_>,
    linear: &mut UploadedLinear,
    state: &LinearState,
    weight_grad: &cuda_core::DeviceBuffer<f32>,
    bias_grad: &cuda_core::DeviceBuffer<f32>,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    if reuse_muon_amax {
        materializer.muon_precomputed(&mut linear.weight, &state.weight_muon, weight_grad)?;
    } else {
        materializer.muon(&mut linear.weight, &state.weight_muon)?;
    }
    materializer.adam(&mut linear.bias, &state.bias, bias_grad)
}

fn materialize_evaluation_next_latent(
    materializer: &mut Materializer<'_>,
    next_latent: &mut UploadedNextLat,
    state: &NextLatState,
) -> Result<(), DriverError> {
    materialize_evaluation_layer_norm(materializer, &mut next_latent.norm, &state.norm)?;
    materialize_evaluation_linear(
        materializer,
        &mut next_latent.input_projection,
        &state.input_projection,
    )?;
    materialize_evaluation_linear(materializer, &mut next_latent.transition, &state.transition)?;
    materialize_evaluation_linear(
        materializer,
        &mut next_latent.output_projection,
        &state.output_projection,
    )
}

fn materialize_evaluation_block(
    materializer: &mut Materializer<'_>,
    block_index: usize,
    block: &mut crate::upload::UploadedBlock,
    state: &BlockState,
) -> Result<(), DriverError> {
    materialize_evaluation_layer_norm(materializer, &mut block.ln_1, &state.ln_1)?;
    if gpt2_nvfp4::canon_ac_enabled() {
        materializer.fp32_master(&mut block.canon_a, &state.canon_a.x_master)?;
    }
    materialize_evaluation_linear(materializer, &mut block.attn_qkv, &state.attn_qkv)?;
    if gpt2_nvfp4::uses_full_attention(block_index) {
        materializer.master(&mut block.attn_qk_scale, &state.attn_qk_scale.x_master)?;
    }
    materialize_evaluation_linear(materializer, &mut block.attn_c_proj, &state.attn_c_proj)?;
    materialize_evaluation_layer_norm(materializer, &mut block.ln_2, &state.ln_2)?;
    if gpt2_nvfp4::canon_ac_enabled() {
        materializer.fp32_master(&mut block.canon_c, &state.canon_c.x_master)?;
    }
    materialize_evaluation_linear(materializer, &mut block.mlp_up, &state.mlp_up)?;
    materialize_evaluation_linear(materializer, &mut block.mlp_down, &state.mlp_down)
}

fn materialize_evaluation_layer_norm(
    materializer: &mut Materializer<'_>,
    layer_norm: &mut UploadedLayerNorm,
    state: &LayerNormState,
) -> Result<(), DriverError> {
    materializer.master(&mut layer_norm.weight, &state.weight.x_master)?;
    materializer.master(&mut layer_norm.bias, &state.bias.x_master)
}

fn materialize_evaluation_linear(
    materializer: &mut Materializer<'_>,
    linear: &mut UploadedLinear,
    state: &LinearState,
) -> Result<(), DriverError> {
    materializer.symexp_lin_muon_master(
        &mut linear.weight,
        &state.weight_muon.x_master,
        &state.weight_muon.symexp_lin,
    )?;
    materializer.master(&mut linear.bias, &state.bias.x_master)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppResult, training::Trainer};
    #[test]
    #[ignore = "requires full approximately 1B GPU initialization; no training"]
    fn linear_bias_preserves_captured_2530_state_without_exponential_overflow() -> AppResult {
        let mut trainer = Trainer::new(0x4750_5432)?;
        let stream = trainer.runtime.stream.as_ref();
        // Exact extrema at the same coordinate in the first-failure state dump.
        let index = 4117;
        let z = 1.713539362_f32;
        let x = 1.016125321_f32;
        let bias = &trainer.buffers.optimizer_state.blocks[3].attn_qkv.bias;
        for (buffer, value) in [(&bias.z_master, z), (&bias.x_master, x)] {
            unsafe {
                cuda_core::memory::memcpy_htod_async(
                    buffer.cu_deviceptr() + (index * size_of::<f32>()) as u64,
                    &value as *const f32,
                    size_of::<f32>(),
                    stream.cu_stream(),
                )?;
            }
            stream.synchronize()?;
        }
        trainer.buffers.optimizer_state.step = 2513;
        let beta = crate::training::learning_rate::schedule_free_beta(2514);
        let expected = z + beta * (x - z);
        let old_effective = ((12.5 * expected).exp() - 1.0 + expected) / 12.5;
        assert!(old_effective > 65504.0);

        fn decoded_bias(trainer: &Trainer, index: usize) -> AppResult<f32> {
            let t = &trainer.uploaded.blocks[3].attn_qkv.bias;
            let stream = trainer.runtime.stream.as_ref();
            let bytes = t.bytes.to_host_vec(stream)?;
            let scales = t.scales.to_host_vec(stream)?;
            let global = t.global_scale.to_host_vec(stream)?[0];
            let code = rust_kernels_cuda::nvfp4::csf::scale_at(&scales, index / 16);
            let exponent = (code >> 3) & 15;
            let mantissa = code & 7;
            let scale = if exponent == 0 {
                mantissa as f32 * 2.0_f32.powi(-9)
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
            };
            let nibble = (bytes[index / 2] >> ((index % 2) * 4)) & 15;
            let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(nibble & 7) as usize];
            Ok(value * if nibble & 8 != 0 { -scale } else { scale } * global)
        }
        training_bias(&mut trainer, beta, false)?;
        let actual = decoded_bias(&trainer, index)?;
        assert!(
            (actual - expected).abs() < 0.15,
            "captured state must stay in ordinary bias units: actual={actual}, expected={expected}"
        );
        evaluation_bias(&mut trainer)?;
        assert!((decoded_bias(&trainer, index)? - x).abs() < 0.15);
        // The normal post-update producer supplies this maximum. Seed valid cache
        // metadata as well as raw masters when testing the cached restore path.
        let amax = &trainer.buffers.optimizer_state.blocks[3]
            .attn_qkv
            .bias
            .schedule_amax;
        let stream = trainer.runtime.stream.as_ref();
        unsafe {
            cuda_core::memory::memcpy_htod_async(
                amax.cu_deviceptr(),
                &expected as *const f32,
                size_of::<f32>(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        training_bias(&mut trainer, beta, true)?;
        assert_eq!(decoded_bias(&trainer, index)?.to_bits(), actual.to_bits());
        Ok(())
    }

    fn training_bias(trainer: &mut Trainer, beta: f32, reuse_amax: bool) -> AppResult {
        let stream = trainer.runtime.stream.as_ref();
        let mut m = Materializer::new(
            stream,
            &trainer.runtime.optimizer,
            &trainer.runtime.f32_ops,
            &mut trainer.buffers.optimizer,
            beta,
            crate::training::symexp_lin::beta(),
            reuse_amax,
            false,
        );
        let grads = &trainer.buffers.backward.blocks[3];
        materialize_linear(
            &mut m,
            &mut trainer.uploaded.blocks[3].attn_qkv,
            &trainer.buffers.optimizer_state.blocks[3].attn_qkv,
            &grads.d_attn_qkv_weight,
            &grads.d_attn_qkv_bias,
            false,
        )?;
        Ok(())
    }
    fn evaluation_bias(trainer: &mut Trainer) -> AppResult {
        let stream = trainer.runtime.stream.as_ref();
        let mut m = Materializer::new(
            stream,
            &trainer.runtime.optimizer,
            &trainer.runtime.f32_ops,
            &mut trainer.buffers.optimizer,
            0.0,
            crate::training::symexp_lin::beta(),
            false,
            false,
        );
        materialize_evaluation_linear(
            &mut m,
            &mut trainer.uploaded.blocks[3].attn_qkv,
            &trainer.buffers.optimizer_state.blocks[3].attn_qkv,
        )?;
        Ok(())
    }
}
