use cuda_core::DriverError;
use gpt2_nvfp4::{GPT2_EMBEDDING_DIM, NEXTLAT_HIDDEN_DIM};
use rust_kernels_cuda::next_latent::{NextLatResidualAddArgs, NextLatSmoothL1Args};

use super::super::forward::NextLatForwardArgs;
use super::project_affine_tma;

pub(in crate::training::next_latent) fn output_and_loss(
    args: NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    project_affine_tma(
        args.stream,
        args.tma,
        args.tma_scale_pack,
        &mut args.buffers.tma_descriptors,
        &mut args.buffers.tma_input_scale_packed,
        &mut args.buffers.tma_weight_scale_packed,
        args.buffers.act2_quant.rowwise(),
        args.weights.output_projection.weight.device(),
        args.weights.output_projection.bias.device(),
        &mut args.buffers.act2,
        args.row_count,
        NEXTLAT_HIDDEN_DIM,
        GPT2_EMBEDDING_DIM,
    )?;
    args.next_latent.residual_add(NextLatResidualAddArgs {
        stream: args.stream,
        delta: &args.buffers.act2,
        residual: args.current_states,
        out: &mut args.buffers.next_token_embeddings,
        len: args.row_count * GPT2_EMBEDDING_DIM,
    })?;
    args.next_latent.smooth_l1(NextLatSmoothL1Args {
        stream: args.stream,
        predicted_next_states: &args.buffers.next_token_embeddings,
        target_states: args.current_states,
        losses: &mut args.buffers.losses,
        d_predicted_next_states: &mut args.buffers.d_predicted,
        batch_size: args.batch_size,
        seq_len: args.seq_len,
        embedding_dim: GPT2_EMBEDDING_DIM,
        lambda: args.lambda,
    })
}
