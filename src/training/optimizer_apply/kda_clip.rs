use cuda_core::{CudaStream, DriverError};
use gpt2_nvfp4::{AttentionDims, GPT2_TOKEN_ROWS_U32, uses_full_attention};
use rust_kernels_cuda::optimizer::KdaMuonClipArgs;

use crate::training::runtime::Runtime;
use crate::upload::UploadedModel;

use super::super::OptimizerTrace;
use super::super::optimizer::OptimizerScratch;
use super::super::optimizer_state::OptimizerStateBuffers;
use super::super::tape::ForwardTapeBuffers;
use super::timed_ms;

const KDA_QK_CLIP_TAU: f32 = 100.0;

pub(super) fn apply_kda_muon_clip(
    stream: &CudaStream,
    runtime: &Runtime,
    uploaded: &mut UploadedModel,
    tape: &ForwardTapeBuffers,
    scratch: &mut OptimizerScratch,
    state: &mut OptimizerStateBuffers,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.kda_clip_ms += timed_ms(|| {
        for block_index in 0..uploaded.blocks.len() {
            let full_attention = uses_full_attention(block_index);
            let dims = AttentionDims::new(full_attention);
            let block = &mut uploaded.blocks[block_index];
            let qkv_state = &mut state.blocks[block_index].attn_qkv.weight_muon;
            runtime.optimizer.apply_kda_muon_clip(KdaMuonClipArgs {
                stream,
                qkv: tape.block_qkv(block_index),
                bytes: &mut block.attn_qkv.weight.bytes,
                scales: &mut block.attn_qkv.weight.scales,
                global_scale: &mut block.attn_qkv.weight.global_scale,
                z_master: &mut qkv_state.z_master,
                x_master: &mut qkv_state.x_master,
                momentum: &mut qkv_state.momentum,
                scores: &mut scratch.kda_clip_scores,
                amax: &mut scratch.amax,
                chunk_amax: &mut scratch.chunk_amax,
                row_count: GPT2_TOKEN_ROWS_U32,
                qkv_dim: dims.qkv_dim,
                input_dim: dims.embedding_dim,
                embedding_dim: dims.embedding_dim,
                head_count: dims.head_count,
                head_dim: dims.head_dim,
                tau: KDA_QK_CLIP_TAU,
                silu_qk: (!full_attention) as u32,
            })?;
        }
        Ok(())
    })?;
    Ok(())
}
