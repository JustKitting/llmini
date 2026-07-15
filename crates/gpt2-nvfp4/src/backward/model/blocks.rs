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
    pub attention_scratch: &'ctx mut BlockAttentionBackwardScratch<'scratch>,
    pub mlp_scratch: &'ctx mut MlpBackwardScratch<'scratch>,
    pub seeds: Gpt2BackwardSeeds,
}

pub(super) fn run_blocks(args: BlocksBackwardRun<'_, '_, '_, '_>) -> Result<(), DriverError> {
    for block_index in (0..GPT2_N_LAYER).rev() {
        let (lower_blocks, current_and_after) = args.blocks.split_at_mut(block_index);
        let current = &mut current_and_after[0];
        let d_residual_in = if block_index == 0 {
            &mut *args.d_embedding_residual
        } else {
            &mut *lower_blocks[block_index - 1].d_residual_out
        };
        run_block(
            args.stream,
            args.modules,
            args.saved,
            args.weights,
            current,
            d_residual_in,
            &mut *args.attention_scratch,
            &mut *args.mlp_scratch,
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
    d_residual_in: &mut DeviceBuffer<f32>,
    attention_scratch: &mut BlockAttentionBackwardScratch<'scratch>,
    mlp_scratch: &mut MlpBackwardScratch<'scratch>,
    seeds: Gpt2BackwardSeeds,
    block_index: usize,
) -> Result<(), DriverError> {
    let mut grads = current.reborrow_with_residual_in(d_residual_in);
    let use_full_attention = uses_full_attention(block_index);
    mlp_side_backward(BlockMlpBackwardArgs {
        stream,
        modules: modules.mlp,
        saved: saved.blocks[block_index],
        ln_2: weights.block_ln_2[block_index],
        mlp_projections: weights.mlp[block_index],
        grads: grads.reborrow(),
        scratch: mlp_scratch.reborrow(),
        seeds: seeds.mlp[block_index],
    })?;
    attention_side_backward(BlockAttentionBackwardArgs {
        use_full_attention,
        reuse_forward_probs: use_full_attention && block_index == GPT2_N_LAYER - 1,
        stream,
        modules: modules.attention,
        saved: saved.blocks[block_index],
        ln_1: weights.block_ln_1[block_index],
        projections: weights.attention[block_index],
        grads: grads.reborrow(),
        scratch: attention_scratch.reborrow(),
        seeds: seeds.attention[block_index],
    })
}
