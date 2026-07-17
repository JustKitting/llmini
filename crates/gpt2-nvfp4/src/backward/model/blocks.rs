use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use super::types::{Gpt2BackwardModules, Gpt2BackwardSeeds, Gpt2BackwardWeights};
use crate::backward::{
    BlockAttentionBackwardArgs, BlockAttentionBackwardScratch, BlockMlpBackwardArgs,
    MlpBackwardScratch, attention_side_backward, mlp_side_backward,
};
use crate::types::{BlockBackwardGrads, Gpt2ForwardSaved};
use crate::{GPT2_N_LAYER, uses_full_attention};

pub(super) struct BlocksBackwardRun<'ctx, 'a, 'scratch, 'out> {
    pub stream: &'a CudaStream,
    pub modules: Gpt2BackwardModules<'a>,
    pub saved: Gpt2ForwardSaved<'a>,
    pub weights: Gpt2BackwardWeights<'a>,
    pub blocks: &'ctx mut [BlockBackwardGrads<'out>; GPT2_N_LAYER],
    pub d_embedding_residual: &'ctx mut DeviceBuffer<f32>,
    pub d_residual_after_attention: &'ctx mut DeviceBuffer<f32>,
    pub d_hidden: &'ctx mut DeviceBuffer<f32>,
    pub d_qkv: &'ctx mut DeviceBuffer<f32>,
    pub d_value_residual: &'ctx mut DeviceBuffer<f32>,
    pub d_mlp_up: &'ctx mut DeviceBuffer<f32>,
    pub attention_scratch: &'ctx mut BlockAttentionBackwardScratch<'scratch>,
    pub mlp_scratch: &'ctx mut MlpBackwardScratch<'scratch>,
    pub initial_d_residual_amax_chunks: u32,
    pub seeds: Gpt2BackwardSeeds,
}

pub(super) fn run_blocks(args: BlocksBackwardRun<'_, '_, '_, '_>) -> Result<(), DriverError> {
    let mut d_residual_amax_chunks = Some(args.initial_d_residual_amax_chunks);
    for block_index in (0..GPT2_N_LAYER).rev() {
        let current = &mut args.blocks[block_index];
        d_residual_amax_chunks = run_block(
            args.stream,
            args.modules,
            args.saved,
            args.weights,
            current,
            &mut *args.d_embedding_residual,
            &mut *args.d_residual_after_attention,
            &mut *args.d_hidden,
            &mut *args.d_qkv,
            &mut *args.d_value_residual,
            &mut *args.d_mlp_up,
            &mut *args.attention_scratch,
            &mut *args.mlp_scratch,
            d_residual_amax_chunks,
            args.seeds,
            block_index,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_block<'a, 'scratch, 'out>(
    stream: &'a cuda_core::CudaStream,
    modules: Gpt2BackwardModules<'a>,
    saved: Gpt2ForwardSaved<'a>,
    weights: Gpt2BackwardWeights<'a>,
    current: &mut BlockBackwardGrads<'out>,
    d_residual: &mut DeviceBuffer<f32>,
    d_residual_after_attention: &mut DeviceBuffer<f32>,
    d_hidden: &mut DeviceBuffer<f32>,
    d_qkv: &mut DeviceBuffer<f32>,
    d_value_residual: &mut DeviceBuffer<f32>,
    d_mlp_up: &mut DeviceBuffer<f32>,
    attention_scratch: &mut BlockAttentionBackwardScratch<'scratch>,
    mlp_scratch: &mut MlpBackwardScratch<'scratch>,
    precomputed_d_residual_amax_chunks: Option<u32>,
    seeds: Gpt2BackwardSeeds,
    block_index: usize,
) -> Result<Option<u32>, DriverError> {
    let mut grads = current.reborrow();
    let use_full_attention = uses_full_attention(block_index);
    let d_residual_after_attention_amax_chunks = mlp_side_backward(BlockMlpBackwardArgs {
        block_index,
        stream,
        modules: modules.mlp,
        saved: saved.blocks[block_index],
        ln_2: weights.block_ln_2[block_index],
        mlp_projections: weights.mlp[block_index],
        d_residual_out: &*d_residual,
        precomputed_d_residual_amax_chunks,
        d_residual_after_attention: &mut *d_residual_after_attention,
        d_residual_after_attention_chunk_amax: &mut *attention_scratch.c_proj.linear.e_h.chunk_amax,
        d_hidden: &mut *d_hidden,
        d_mlp_up: &mut *d_mlp_up,
        grads: grads.reborrow(),
        scratch: mlp_scratch.reborrow(),
        seeds: seeds.mlp[block_index],
    })?;
    attention_side_backward(BlockAttentionBackwardArgs {
        block_index,
        use_full_attention,
        reuse_forward_probs: use_full_attention,
        stream,
        modules: modules.attention,
        saved: saved.blocks[block_index],
        ln_1: weights.block_ln_1[block_index],
        projections: weights.attention[block_index],
        d_residual_after_attention: &*d_residual_after_attention,
        precomputed_d_residual_after_attention_amax_chunks: Some(
            d_residual_after_attention_amax_chunks,
        ),
        d_residual_in: d_residual,
        d_residual_in_chunk_amax: &mut *mlp_scratch.down_linear.e_h.chunk_amax,
        d_hidden,
        d_qkv,
        d_value_residual,
        grads: grads.reborrow(),
        scratch: attention_scratch.reborrow(),
        seeds: seeds.attention[block_index],
    })
}
