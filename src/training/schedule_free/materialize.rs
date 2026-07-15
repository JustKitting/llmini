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
    let mut materializer = Materializer::new(stream, &runtime.optimizer, scratch, beta);

    materializer.adam(&mut uploaded.token_embedding, &state.token_embedding)?;
    materialize_layer_norm(&mut materializer, &mut uploaded.ln_f, &state.ln_f)?;
    materialize_next_latent(
        &mut materializer,
        &mut uploaded.next_latent,
        &state.next_latent,
    )?;

    for (block, state) in uploaded.blocks.iter_mut().zip(state.blocks.iter()) {
        materialize_block(&mut materializer, block, state)?;
    }

    Ok(())
}

fn materialize_next_latent(
    materializer: &mut Materializer<'_>,
    next_latent: &mut UploadedNextLat,
    state: &NextLatState,
) -> Result<(), DriverError> {
    materialize_layer_norm(materializer, &mut next_latent.norm, &state.norm)?;
    materialize_linear(
        materializer,
        &mut next_latent.input_projection,
        &state.input_projection,
    )?;
    materialize_linear(materializer, &mut next_latent.transition, &state.transition)?;
    materialize_linear(
        materializer,
        &mut next_latent.output_projection,
        &state.output_projection,
    )
}

fn materialize_block(
    materializer: &mut Materializer<'_>,
    block: &mut crate::upload::UploadedBlock,
    state: &BlockState,
) -> Result<(), DriverError> {
    materialize_layer_norm(materializer, &mut block.ln_1, &state.ln_1)?;
    materialize_linear(materializer, &mut block.attn_qkv, &state.attn_qkv)?;
    materialize_linear(materializer, &mut block.attn_c_proj, &state.attn_c_proj)?;
    materialize_layer_norm(materializer, &mut block.ln_2, &state.ln_2)?;
    materialize_linear(materializer, &mut block.mlp_up, &state.mlp_up)?;
    materialize_linear(materializer, &mut block.mlp_down, &state.mlp_down)
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
) -> Result<(), DriverError> {
    materializer.muon(&mut linear.weight, &state.weight_muon)?;
    materializer.adam(&mut linear.bias, &state.bias)
}
