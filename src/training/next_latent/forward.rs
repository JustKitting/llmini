use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_EMBEDDING_DIM, GPT2_LAYER_NORM_EPSILON, NEXTLAT_INPUT_DIM};
use rust_kernels_cuda::embedding::{EmbeddingArgs, EmbeddingModule};
use rust_kernels_cuda::layer_norm::{GptLayerNormArgs, LayerNormModule};
use rust_kernels_cuda::next_latent::{NextLatConcatArgs, NextLatModule};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
};

use super::buffers::NextLatBuffers;
use super::projection::{output_and_loss, projection_gelu1, projection_gelu2};
use super::quantize::quantize_input;
use crate::upload::UploadedNextLat;

pub struct NextLatForwardArgs<'a, 'out> {
    pub stream: &'a CudaStream,
    pub embedding: &'a EmbeddingModule,
    pub layer_norm: &'a LayerNormModule,
    pub quant: &'a Nvfp4QuantModule,
    pub tma: &'a Nvfp4GemmModule,
    pub tma_scale_pack: &'a Sm120ScalePackModule,
    pub next_latent: &'a NextLatModule,
    pub token_embedding: Nvfp4DeviceTensor<'a>,
    pub weights: &'a UploadedNextLat,
    pub targets: &'a DeviceBuffer<u32>,
    pub current_states: &'a DeviceBuffer<f32>,
    pub buffers: &'out mut NextLatBuffers,
    pub batch_size: u32,
    pub seq_len: u32,
    pub row_count: u32,
    pub lambda: f32,
}

pub fn forward(mut args: NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    lookup_next_tokens(&mut args)?;
    args.next_latent.concat_input(NextLatConcatArgs {
        stream: args.stream,
        next_token_embeddings: &args.buffers.next_token_embeddings,
        current_states: args.current_states,
        out: &mut args.buffers.concat,
        row_count: args.row_count,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })?;
    norm_and_quantize_input(&mut args)?;
    projection_gelu1(&mut args)?;
    projection_gelu2(&mut args)?;
    output_and_loss(args)
}

fn lookup_next_tokens(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    args.embedding.token_embedding(EmbeddingArgs {
        stream: args.stream,
        tokens: args.targets,
        token_embedding: args.token_embedding,
        residual: &mut args.buffers.next_token_embeddings,
        hidden_len: args.row_count * GPT2_EMBEDDING_DIM,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })
}

fn norm_and_quantize_input(args: &mut NextLatForwardArgs<'_, '_>) -> Result<(), DriverError> {
    args.layer_norm.gpt_layer_norm(GptLayerNormArgs {
        stream: args.stream,
        residual: &args.buffers.concat,
        weight: args.weights.norm.weight.device(),
        bias: args.weights.norm.bias.device(),
        normalized: &mut args.buffers.normalized,
        normalized_amax: &mut args.buffers.normalized_amax,
        mean: &mut args.buffers.mean,
        inv_std: &mut args.buffers.inv_std,
        row_count: args.row_count,
        embedding_dim: NEXTLAT_INPUT_DIM,
        epsilon: GPT2_LAYER_NORM_EPSILON,
        output_scale: 1.0,
    })?;
    quantize_input(args)
}
