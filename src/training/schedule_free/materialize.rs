use cuda_core::{CudaStream, DriverError};

use super::tensor::Materializer;
use crate::training::runtime::Runtime;
use crate::upload::{UploadedLayerNorm, UploadedLinear, UploadedModel, UploadedNextLat};

use super::super::learning_rate::schedule_free_beta;
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
) -> Result<(), DriverError> {
    let beta = schedule_free_beta(state.next_step());
    let reuse_muon_amax = state.next_step() > 1;
    let mut materializer = Materializer::new(stream, &runtime.optimizer, scratch, beta);

    materializer.adam(&mut uploaded.token_embedding, &state.token_embedding)?;
    materialize_layer_norm(&mut materializer, &mut uploaded.ln_f, &state.ln_f)?;
    materialize_next_latent(
        &mut materializer,
        &mut uploaded.next_latent,
        &state.next_latent,
        reuse_muon_amax,
    )?;

    for (block, state) in uploaded.blocks.iter_mut().zip(state.blocks.iter()) {
        materialize_block(&mut materializer, block, state, reuse_muon_amax)?;
    }

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
    let mut materializer = Materializer::new(stream, &runtime.optimizer, scratch, 0.0);

    materializer.master(
        &mut uploaded.token_embedding,
        &state.token_embedding.x_master,
    )?;
    materialize_evaluation_layer_norm(&mut materializer, &mut uploaded.ln_f, &state.ln_f)?;
    materialize_evaluation_next_latent(
        &mut materializer,
        &mut uploaded.next_latent,
        &state.next_latent,
    )?;

    for (block, state) in uploaded.blocks.iter_mut().zip(state.blocks.iter()) {
        materialize_evaluation_block(&mut materializer, block, state)?;
    }

    Ok(())
}

fn materialize_next_latent(
    materializer: &mut Materializer<'_>,
    next_latent: &mut UploadedNextLat,
    state: &NextLatState,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    materialize_layer_norm(materializer, &mut next_latent.norm, &state.norm)?;
    materialize_linear(
        materializer,
        &mut next_latent.input_projection,
        &state.input_projection,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut next_latent.transition,
        &state.transition,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut next_latent.output_projection,
        &state.output_projection,
        reuse_muon_amax,
    )
}

fn materialize_block(
    materializer: &mut Materializer<'_>,
    block: &mut crate::upload::UploadedBlock,
    state: &BlockState,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    materialize_layer_norm(materializer, &mut block.ln_1, &state.ln_1)?;
    materialize_linear(
        materializer,
        &mut block.attn_qkv,
        &state.attn_qkv,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut block.attn_c_proj,
        &state.attn_c_proj,
        reuse_muon_amax,
    )?;
    materialize_layer_norm(materializer, &mut block.ln_2, &state.ln_2)?;
    materialize_linear(
        materializer,
        &mut block.mlp_up,
        &state.mlp_up,
        reuse_muon_amax,
    )?;
    materialize_linear(
        materializer,
        &mut block.mlp_down,
        &state.mlp_down,
        reuse_muon_amax,
    )
}

fn materialize_layer_norm(
    materializer: &mut Materializer<'_>,
    layer_norm: &mut UploadedLayerNorm,
    state: &LayerNormState,
) -> Result<(), DriverError> {
    materializer.adam(&mut layer_norm.weight, &state.weight)?;
    materializer.adam(&mut layer_norm.bias, &state.bias)
}

fn materialize_linear(
    materializer: &mut Materializer<'_>,
    linear: &mut UploadedLinear,
    state: &LinearState,
    reuse_muon_amax: bool,
) -> Result<(), DriverError> {
    if reuse_muon_amax {
        materializer.muon_precomputed(&mut linear.weight, &state.weight_muon)?;
    } else {
        materializer.muon(&mut linear.weight, &state.weight_muon)?;
    }
    materializer.adam(&mut linear.bias, &state.bias)
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
    block: &mut crate::upload::UploadedBlock,
    state: &BlockState,
) -> Result<(), DriverError> {
    materialize_evaluation_layer_norm(materializer, &mut block.ln_1, &state.ln_1)?;
    materialize_evaluation_linear(materializer, &mut block.attn_qkv, &state.attn_qkv)?;
    materialize_evaluation_linear(materializer, &mut block.attn_c_proj, &state.attn_c_proj)?;
    materialize_evaluation_layer_norm(materializer, &mut block.ln_2, &state.ln_2)?;
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
    materializer.master(&mut linear.weight, &state.weight_muon.x_master)?;
    materializer.master(&mut linear.bias, &state.bias.x_master)
}
