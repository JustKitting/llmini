use cuda_core::{DriverError, memory};

use super::blocks::{BlocksBackwardRun, run_blocks};
use super::final_head::run_final_head;
use super::types::{Gpt2BackwardArgs, Gpt2BackwardInput};
use crate::GPT2_EMBEDDING_DIM;
use crate::backward::{Gpt2LayerNormBackwardAmaxArgs, layer_norm_backward_amax};
use crate::types::Gpt2BackwardGrads;
use rust_kernels_cuda::residual::ResidualGradAccumulateArgs;

pub fn backward(args: Gpt2BackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    backward_with_hook(args, &mut |_, _| Ok(false))
}

pub fn backward_with_hook(
    args: Gpt2BackwardArgs<'_, '_, '_>,
    hook: &mut dyn FnMut(usize, &mut cuda_core::DeviceBuffer<f32>) -> Result<bool, DriverError>,
) -> Result<(), DriverError> {
    backward_with_boundaries(args, hook, &mut |_, _, _| Ok(false))
}

pub fn backward_with_boundaries(
    args: Gpt2BackwardArgs<'_, '_, '_>,
    hook: &mut dyn FnMut(usize, &mut cuda_core::DeviceBuffer<f32>) -> Result<bool, DriverError>,
    boundary: &mut dyn FnMut(
        usize,
        bool,
        &mut cuda_core::DeviceBuffer<f32>,
    ) -> Result<bool, DriverError>,
) -> Result<(), DriverError> {
    let Gpt2BackwardArgs {
        stream,
        modules,
        input,
        weights,
        grads,
        scratch,
        seeds,
    } = args;
    let mut attention_scratch = scratch.attention;
    let mut mlp_scratch = scratch.mlp;
    let d_residual_after_attention = scratch.d_residual_after_attention;
    let d_hidden = scratch.d_hidden;
    let d_qkv = scratch.d_qkv;
    let d_value_residual = scratch.d_value_residual;
    let d_mlp_up = scratch.d_mlp_up;
    let Gpt2BackwardGrads {
        dlogits,
        d_embedding_residual,
        d_xsa_alphas,
        mut blocks,
        mut final_norm,
    } = grads;

    let (saved, initial_d_residual_amax_chunks) = match input {
        Gpt2BackwardInput::TokenLoss { saved, targets, losses, extra_final_normalized_grad, d_lm_head_weight } => {
            run_final_head(
                stream, modules, saved, weights, targets, losses, dlogits, &mut *d_hidden,
                d_lm_head_weight, scratch.final_head, seeds.final_head,
            )?;
            if let Some(extra) = extra_final_normalized_grad {
                modules.residual.grad_accumulate(ResidualGradAccumulateArgs {
                    stream,
                    branch: extra,
                    out: &mut *d_hidden,
                    len: saved.row_count * GPT2_EMBEDDING_DIM,
                })?;
            }
            let chunks = layer_norm_backward_amax(Gpt2LayerNormBackwardAmaxArgs {
                stream,
                module: modules.final_norm,
                weights: weights.ln_f,
                saved: saved.final_norm,
                grads: final_norm.reborrow(),
                d_normalized: &*d_hidden,
                d_residual: &mut *d_embedding_residual,
                chunk_amax: &mut *mlp_scratch.down_linear.e_h.chunk_amax,
                output_scale: 1.0,
            })?;
            (saved.blocks, Some(chunks))
        }
        Gpt2BackwardInput::Residual { saved, output_gradient } => {
            let len = (saved[0].row_count as usize) * (GPT2_EMBEDDING_DIM as usize);
            if len == 0 || output_gradient.len() != len || d_embedding_residual.len() < len {
                return Err(DriverError(1));
            }
            stream.context().bind_to_thread()?;
            unsafe {
                memory::memcpy_dtod_async(
                    d_embedding_residual.cu_deviceptr(),
                    output_gradient.cu_deviceptr(),
                    len * size_of::<f32>(),
                    stream.cu_stream(),
                )?;
            }
            (saved, None)
        }
    };
    let head_changed = boundary(crate::GPT2_N_LAYER * 2, true, d_embedding_residual)?;
    run_blocks(
        BlocksBackwardRun {
            stream,
            modules,
            saved,
            weights,
            blocks: &mut blocks,
            d_xsa_alphas,
            d_embedding_residual,
            d_residual_after_attention,
            d_hidden,
            d_qkv,
            d_value_residual,
            d_mlp_up,
            attention_scratch: &mut attention_scratch,
            mlp_scratch: &mut mlp_scratch,
            initial_d_residual_amax_chunks: if head_changed {
                None
            } else {
                initial_d_residual_amax_chunks
            },
            seeds,
        },
        hook,
        boundary,
    )
}
