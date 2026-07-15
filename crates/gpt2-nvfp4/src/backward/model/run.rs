use cuda_core::DriverError;

use super::blocks::{BlocksBackwardRun, run_blocks};
use super::final_head::run_final_head;
use super::types::Gpt2BackwardArgs;
use crate::GPT2_EMBEDDING_DIM;
use crate::backward::{Gpt2LayerNormBackwardArgs, layer_norm_backward};
use crate::types::Gpt2BackwardGrads;
use rust_kernels_cuda::residual::ResidualGradAccumulateArgs;

pub fn backward(args: Gpt2BackwardArgs<'_, '_, '_>) -> Result<(), DriverError> {
    let Gpt2BackwardArgs {
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
        extra_final_normalized_grad,
        d_lm_head_weight,
        grads,
        scratch,
        seeds,
    } = args;
    let mut attention_scratch = scratch.attention;
    let mut mlp_scratch = scratch.mlp;
    let d_residual_after_attention = scratch.d_residual_after_attention;
    let d_hidden = scratch.d_hidden;
    let d_qkv = scratch.d_qkv;
    let d_mlp_up = scratch.d_mlp_up;
    let d_mlp_relu2 = scratch.d_mlp_relu2;
    let Gpt2BackwardGrads {
        dlogits,
        d_embedding_residual,
        mut blocks,
        mut final_norm,
    } = grads;

    run_final_head(
        stream,
        modules,
        saved,
        weights,
        targets,
        losses,
        dlogits,
        &mut *d_hidden,
        d_lm_head_weight,
        scratch.final_head,
        seeds.final_head,
    )?;
    if let Some(extra) = extra_final_normalized_grad {
        modules
            .residual
            .grad_accumulate(ResidualGradAccumulateArgs {
                stream,
                branch: extra,
                out: &mut *d_hidden,
                len: saved.row_count * GPT2_EMBEDDING_DIM,
            })?;
    }
    layer_norm_backward(Gpt2LayerNormBackwardArgs {
        stream,
        module: modules.final_norm,
        weights: weights.ln_f,
        saved: saved.final_norm,
        grads: final_norm.reborrow(),
        d_normalized: &*d_hidden,
        d_residual: &mut *d_embedding_residual,
    })?;
    run_blocks(BlocksBackwardRun {
        stream,
        modules,
        saved,
        weights,
        blocks: &mut blocks,
        d_embedding_residual,
        d_residual_after_attention,
        d_hidden,
        d_qkv,
        d_mlp_up,
        d_mlp_relu2,
        attention_scratch: &mut attention_scratch,
        mlp_scratch: &mut mlp_scratch,
        seeds,
    })
}
