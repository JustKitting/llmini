use cuda_core::DriverError;
use gpt2_nvfp4::{NEXTLAT_HIDDEN_DIM, NEXTLAT_INPUT_DIM};
use rust_kernels_cuda::next_latent::NextLatGeluArgs;

use super::super::forward::NextLatForwardArgs;
use super::super::quantize::quantize_activation;
use super::project_affine_tma;

pub(in crate::training::next_latent) fn projection_gelu1(
    args: &mut NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    project_affine_tma(
        args.stream,
        args.tma,
        args.tma_scale_pack,
        &mut args.buffers.tma_descriptors,
        &mut args.buffers.tma_input_scale_packed,
        &mut args.buffers.tma_weight_scale_packed,
        args.buffers.input_quant.rowwise(),
        args.weights.input_projection.weight.device(),
        args.weights.input_projection.bias.device(),
        &mut args.buffers.pre1,
        args.row_count,
        NEXTLAT_INPUT_DIM,
        NEXTLAT_HIDDEN_DIM,
    )?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre1,
        out: &mut args.buffers.act1,
        len: args.row_count * NEXTLAT_HIDDEN_DIM,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        buffers.act1_quantize(),
    )
}

pub(in crate::training::next_latent) fn projection_gelu2(
    args: &mut NextLatForwardArgs<'_, '_>,
) -> Result<(), DriverError> {
    project_affine_tma(
        args.stream,
        args.tma,
        args.tma_scale_pack,
        &mut args.buffers.tma_descriptors,
        &mut args.buffers.tma_input_scale_packed,
        &mut args.buffers.tma_weight_scale_packed,
        args.buffers.act1_quant.rowwise(),
        args.weights.transition.weight.device(),
        args.weights.transition.bias.device(),
        &mut args.buffers.pre2,
        args.row_count,
        NEXTLAT_HIDDEN_DIM,
        NEXTLAT_HIDDEN_DIM,
    )?;
    args.next_latent.gelu(NextLatGeluArgs {
        stream: args.stream,
        input: &args.buffers.pre2,
        out: &mut args.buffers.act2,
        len: args.row_count * NEXTLAT_HIDDEN_DIM,
    })?;
    let buffers = &mut args.buffers;
    quantize_activation(
        args.quant,
        args.stream,
        args.row_count,
        buffers.act2_quantize(),
    )
}
