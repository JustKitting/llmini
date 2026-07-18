use cuda_core::DriverError;
use rust_kernels_cuda::mlp::MlpBlockTopKRouteArgs;
use rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmRoute;

use super::tensors::MlpForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.scratch.input_nvfp4;
    let mut activation_nvfp4 = args.scratch.activation_nvfp4;
    let mut tape = args.tape;
    let hidden = args.hidden;

    let input = input_nvfp4.quantize_hidden_precomputed(
        args.quant_module,
        &hidden,
        crate::GPT2_EMBEDDING_DIM,
    )?;
    if let Some(tape) = tape.as_mut() {
        tape.save_up_input(hidden.stream, input)?;
    }

    args.tma_scale_pack.pack(
        hidden.stream,
        input.scales,
        args.scratch.tma_input_scale_packed,
        hidden.row_count,
        crate::GPT2_EMBEDDING_DIM,
    )?;
    args.tma_scale_pack.pack(
        hidden.stream,
        args.projections.up.weight_device.scales,
        args.scratch.tma_weight_scale_packed,
        crate::GPT2_MLP_DIM,
        crate::GPT2_EMBEDDING_DIM,
    )?;
    args.tma_module.prepare_tma_nvfp4_device_scales_into(
        hidden.stream,
        input.bytes,
        args.scratch.tma_input_scale_packed,
        args.projections.up.weight_device.bytes,
        args.scratch.tma_weight_scale_packed,
        hidden.row_count,
        crate::GPT2_EMBEDDING_DIM,
        crate::GPT2_MLP_DIM,
        args.scratch.tma_descriptors,
    )?;
    args.tma_module
        .gemm_tma_nvfp4_rowwise_a_scale_relu2_compact(
            hidden.stream,
            args.scratch.tma_descriptors,
            tape.as_mut().map(|tape| &mut *tape.pre_activation_f16),
            args.scratch.activation,
            args.projections.up.bias,
            hidden.row_count,
            crate::GPT2_EMBEDDING_DIM,
            crate::GPT2_MLP_DIM,
            input.global_scales,
            args.projections.up.weight_device.global_scale,
        )?;

    activation_nvfp4.quantize_row_amax(
        args.quant_module,
        hidden.stream,
        args.scratch.activation,
        &mut *hidden.normalized_amax,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
    )?;

    let input = activation_nvfp4.device();
    if let Some(tape) = tape.as_mut() {
        tape.save_down_input(hidden.stream, input)?;
    }
    let mut route_masks = if crate::uses_block_topk_mlp(args.block_index) {
        Some(match tape.as_mut() {
            Some(tape) => &mut *tape.route_masks,
            None => &mut *args.scratch.route_masks,
        })
    } else {
        None
    };
    if let Some(route_masks) = route_masks.as_deref_mut() {
        args.module.block_topk_route(MlpBlockTopKRouteArgs {
            stream: hidden.stream,
            activation: input,
            scores: &mut *args.scratch.route_scores,
            masks: route_masks,
            token_count: hidden.row_count,
            feature_count: crate::GPT2_MLP_DIM,
        })?;
    }

    args.tma_scale_pack.pack(
        hidden.stream,
        input.scales,
        args.scratch.tma_wide_input_scale_packed,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
    )?;
    args.tma_scale_pack.pack(
        hidden.stream,
        args.projections.down.weight_device.scales,
        args.scratch.tma_weight_scale_packed,
        crate::GPT2_EMBEDDING_DIM,
        crate::GPT2_MLP_DIM,
    )?;
    args.tma_module.prepare_tma_nvfp4_device_scales_into(
        hidden.stream,
        input.bytes,
        args.scratch.tma_wide_input_scale_packed,
        args.projections.down.weight_device.bytes,
        args.scratch.tma_weight_scale_packed,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
        crate::GPT2_EMBEDDING_DIM,
        args.scratch.tma_descriptors,
    )?;
    args.tma_module.gemm_tma_nvfp4_rowwise_a_scale_residual(
        hidden.stream,
        args.scratch.tma_descriptors,
        &mut *hidden.residual,
        args.projections.down.bias,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
        crate::GPT2_EMBEDDING_DIM,
        input.global_scales,
        args.projections.down.weight_device.global_scale,
        route_masks.as_deref().map(|masks| {
            Nvfp4GemmRoute::token_m_feature_k(
                masks,
                crate::GPT2_MLP_ROUTE_TOKEN_TILES as u32,
                crate::GPT2_MLP_ROUTE_FEATURE_TILES as u32,
                crate::GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES as u32,
            )
        }),
    )?;

    Ok(hidden)
}
