use cuda_core::{CudaStream, DriverError};
use rust_kernels_cuda::optimizer::OptimizerModule;

use crate::upload::{UploadedBlock, UploadedModel};

use super::super::OptimizerTrace;
use super::super::grad_block::BlockGradBuffers;
use super::super::grads::BackwardBuffers;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::{BlockState, OptimizerStateBuffers};
use super::adam::AdamUpdate;
use super::layer_norm::update_layer_norm_timed;
use super::timed_ms;

pub(super) struct BlockUpdateArgs<'a> {
    pub stream: &'a CudaStream,
    pub optimizer: &'a OptimizerModule,
    pub uploaded: &'a mut UploadedModel,
    pub grads: &'a BackwardBuffers,
    pub scratch: &'a mut OptimizerScratch,
    pub state: &'a mut OptimizerStateBuffers,
    pub step: u32,
    pub average_coefficient: f32,
    pub grad_scale: f32,
    pub trace: &'a mut OptimizerTrace,
}

pub(super) fn update_blocks(args: BlockUpdateArgs<'_>) -> Result<(), DriverError> {
    let mut adam = AdamUpdate::new(
        args.stream,
        args.optimizer,
        args.scratch,
        args.step,
        args.average_coefficient,
        args.grad_scale,
    );
    let blocks_ms = timed_ms(|| {
        for (block_index, ((block, grad), state)) in args
            .uploaded
            .blocks
            .iter_mut()
            .zip(args.grads.blocks.iter())
            .zip(args.state.blocks.iter_mut())
            .enumerate()
        {
            update_block(&mut adam, block_index, block, grad, state, args.trace)?;
        }
        Ok(())
    })?;
    args.trace.blocks_ms = blocks_ms;
    Ok(())
}

pub(super) fn update_block(
    adam: &mut AdamUpdate<'_, '_>,
    block_index: usize,
    block: &mut UploadedBlock,
    grad: &BlockGradBuffers,
    state: &mut BlockState,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.adam_ms += update_layer_norm_timed(adam, &mut block.ln_1, &grad.ln_1, &mut state.ln_1)?;
    trace.adam_ms += adam.update_timed(
        &mut block.attn_qkv.bias,
        &grad.d_attn_qkv_bias,
        &mut state.attn_qkv.bias,
    )?;
    if gpt2_nvfp4::uses_full_attention(block_index) {
        trace.adam_ms += adam.update_timed(
            &mut block.attn_qk_scale,
            &grad.d_attn_qk_scale,
            &mut state.attn_qk_scale,
        )?;
    }
    trace.adam_ms += adam.update_timed(
        &mut block.attn_c_proj.bias,
        &grad.d_attn_c_proj_bias,
        &mut state.attn_c_proj.bias,
    )?;
    trace.adam_ms += update_layer_norm_timed(adam, &mut block.ln_2, &grad.ln_2, &mut state.ln_2)?;
    trace.adam_ms += adam.update_timed(
        &mut block.mlp_up.bias,
        &grad.d_mlp_c_fc_bias,
        &mut state.mlp_up.bias,
    )?;
    trace.adam_ms += adam.update_timed(
        &mut block.mlp_down.bias,
        &grad.d_mlp_c_proj_bias,
        &mut state.mlp_down.bias,
    )?;
    Ok(())
}
