use cuda_core::DriverError;
use rust_kernels_cuda::mlp::{
    ADAPTIVE_RELU2_PACK_WIDTH, AdaptiveRelu2PackArgs, MlpBlockTopKRouteArgs,
};
use rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmRoute;

use super::tensors::MlpForwardArgs;
use crate::types::HiddenStateDevice;

pub(super) fn forward<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    forward_impl(args, true)
}

pub(super) fn forward_impl<'a, 'scratch>(
    args: MlpForwardArgs<'a, 'scratch>,
    add_residual: bool,
) -> Result<HiddenStateDevice<'a>, DriverError> {
    let mut input_nvfp4 = args.scratch.input_nvfp4;
    let mut activation_nvfp4 = args.scratch.activation_nvfp4;
    let (mut up_tape, mut down_tape, mut pre_activation, mut saved_routes) = match args.tape {
        Some(tape) => (
            Some(tape.up_input_nvfp4),
            Some(tape.down_input_nvfp4),
            Some(tape.pre_activation_f16),
            Some(tape.route_masks),
        ),
        None => (None, None, None, None),
    };
    let hidden = args.hidden;
    let fused_adaptive_relu2 =
        crate::GPT2_ADAPTIVE_RELU2 && crate::GPT2_MLP_DIM == ADAPTIVE_RELU2_PACK_WIDTH;

    if !args.input_prequantized {
        if let Some(tape) = up_tape.as_mut() {
            tape.swap_storage(&mut input_nvfp4);
        }
        input_nvfp4.quantize_hidden_precomputed(
            args.quant_module,
            &hidden,
            crate::GPT2_EMBEDDING_DIM,
        )?;
    }
    if let Some(tape) = up_tape.as_mut() {
        tape.swap_storage(&mut input_nvfp4);
    }
    let input = match up_tape.as_ref() {
        Some(tape) => tape.saved(),
        None => input_nvfp4.device(),
    };

    if crate::GPT2_BOUNDED_SWIGLU {
        super::weights::MlpWeights::project_paired(
            args.tma_module,
            hidden.stream,
            input,
            args.projections.up,
            args.scratch.tma_descriptors,
            args.scratch.paired,
            hidden.row_count,
        )?;
        if let Some(observer) = args.projection_observer {
            observer.observe(
                args.block_index,
                hidden.stream,
                args.scratch.paired,
                hidden.row_count as usize * crate::GPT2_MLP_UP,
                false,
            )?;
        }
        args.module.bounded_forward(
            hidden.stream,
            args.scratch.paired,
            args.scratch.activation,
            hidden.row_count,
            crate::GPT2_MLP_DIM,
        )?;
    } else {
        args.tma_module.prepare_tma_nvfp4_csf_weight_into(
            hidden.stream,
            input.bytes,
            input.scales,
            args.projections.up.weight_device.bytes,
            args.projections.up.weight_device.scales,
            hidden.row_count,
            crate::GPT2_EMBEDDING_DIM,
            crate::GPT2_MLP_DIM,
            args.scratch.tma_descriptors,
        )?;
        if crate::GPT2_ADAPTIVE_RELU2 {
            args.tma_module.gemm_tma_nvfp4_rowwise_a_scale_projection(
                hidden.stream,
                args.scratch.tma_descriptors,
                args.scratch.activation,
                args.projections.up.bias,
                hidden.row_count,
                crate::GPT2_EMBEDDING_DIM,
                crate::GPT2_MLP_DIM,
                input.global_scales,
                args.projections.up.weight_device.global_scale,
                None,
                false,
            )?;
            if !fused_adaptive_relu2 {
                args.module.adaptive_relu2_forward(
                    hidden.stream,
                    args.scratch.activation,
                    pre_activation.as_deref_mut(),
                    hidden.row_count,
                    crate::GPT2_MLP_DIM,
                )?;
            }
        } else {
            args.tma_module
                .gemm_tma_nvfp4_rowwise_a_scale_relu2_compact(
                    hidden.stream,
                    args.scratch.tma_descriptors,
                    pre_activation.as_deref_mut(),
                    args.scratch.activation,
                    args.projections.up.bias,
                    hidden.row_count,
                    crate::GPT2_EMBEDDING_DIM,
                    crate::GPT2_MLP_DIM,
                    input.global_scales,
                    args.projections.up.weight_device.global_scale,
                )?;
        }
    }

    if let Some(tape) = down_tape.as_mut() {
        tape.swap_storage(&mut activation_nvfp4);
    }
    if fused_adaptive_relu2 {
        args.module.adaptive_relu2_pack(AdaptiveRelu2PackArgs {
            stream: hidden.stream,
            values: args.scratch.activation,
            tape: pre_activation,
            amax: &mut *hidden.normalized_amax,
            bytes: &mut *activation_nvfp4.bytes,
            scales: &mut *activation_nvfp4.scales,
            globals: &mut *activation_nvfp4.global_scales,
            rows: hidden.row_count,
        })?;
    } else {
        activation_nvfp4.quantize_row_amax(
            args.quant_module,
            hidden.stream,
            args.scratch.activation,
            &mut *hidden.normalized_amax,
            hidden.row_count,
            crate::GPT2_MLP_DIM,
        )?;
    }

    if let Some(tape) = down_tape.as_mut() {
        tape.swap_storage(&mut activation_nvfp4);
    }
    let input = match down_tape.as_ref() {
        Some(tape) => tape.saved(),
        None => activation_nvfp4.device(),
    };
    let mut route_masks = if crate::uses_block_topk_mlp(args.block_index) {
        Some(match saved_routes.as_mut() {
            Some(routes) => &mut **routes,
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
            seq_len: hidden.seq_len,
            feature_count: crate::GPT2_MLP_DIM,
        })?;
    }

    args.tma_module.prepare_tma_nvfp4_csf_weight_into(
        hidden.stream,
        input.bytes,
        input.scales,
        args.projections.down.weight_device.bytes,
        args.projections.down.weight_device.scales,
        hidden.row_count,
        crate::GPT2_MLP_DIM,
        crate::GPT2_EMBEDDING_DIM,
        args.scratch.tma_descriptors,
    )?;
    args.tma_module.gemm_tma_nvfp4_rowwise_a_scale_projection(
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
                hidden.row_count / crate::GPT2_MLP_ROUTE_TILE as u32,
                crate::GPT2_MLP_ROUTE_FEATURE_TILES as u32,
                crate::GPT2_MLP_ROUTE_ACTIVE_FEATURE_TILES as u32,
            )
        }),
        add_residual,
    )?;

    Ok(hidden)
}
