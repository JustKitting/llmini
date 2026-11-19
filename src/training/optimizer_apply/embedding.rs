use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::GPT2_EMBEDDING_DIM;
use rust_kernels_cuda::optimizer::{EmbeddingLookupGradArgs, OptimizerModule};

use super::super::TokenBatch;
use super::super::grads::BackwardBuffers;
use super::super::next_latent::NextLatGradBuffers;

pub(in crate::training) fn add_embedding_lookup_grad(
    stream: &CudaStream,
    optimizer: &OptimizerModule,
    batch: &TokenBatch,
    grads: &mut BackwardBuffers,
    next_latent: &NextLatGradBuffers,
) -> Result<(), DriverError> {
    optimizer.add_embedding_lookup_grad(EmbeddingLookupGradArgs {
        stream,
        tokens: &batch.tokens,
        d_embedding_residual: &grads.d_embedding_residual,
        d_token_embedding: &mut grads.d_lm_head_weight,
        token_count: batch.token_count as u32,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })?;
    optimizer.add_embedding_lookup_grad(EmbeddingLookupGradArgs {
        stream,
        tokens: &batch.targets,
        d_embedding_residual: &next_latent.d_next_token_embeddings,
        d_token_embedding: &mut grads.d_lm_head_weight,
        token_count: batch.token_count as u32,
        embedding_dim: GPT2_EMBEDDING_DIM,
    })
}
