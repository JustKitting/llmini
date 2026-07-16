use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{AttentionDims, GPT2_TOKEN_ROWS_U32, uses_full_attention};
use rust_kernels_cuda::optimizer::KdaMuonClipFactorArgs;

use crate::training::runtime::Runtime;

use super::super::OptimizerTrace;
use super::super::optimizer::OptimizerScratch;
use super::super::tape::ForwardTapeBuffers;
use super::timed_ms;

const KDA_QK_CLIP_TAU: f32 = 100.0;

pub(super) fn prepare_kda_muon_clip_factors(
    stream: &CudaStream,
    runtime: &Runtime,
    tape: &ForwardTapeBuffers,
    qk_norm_max: &DeviceBuffer<f32>,
    scratch: &mut OptimizerScratch,
    trace: &mut OptimizerTrace,
) -> Result<(), DriverError> {
    trace.kda_clip_ms += timed_ms(|| {
        for block_index in 0..gpt2_nvfp4::GPT2_N_LAYER {
            let full_attention = uses_full_attention(block_index);
            let dims = AttentionDims::new(full_attention);
            runtime
                .optimizer
                .prepare_kda_muon_clip_factor(KdaMuonClipFactorArgs {
                    stream,
                    qkv: tape.block_qkv(block_index),
                    qk_norm_max,
                    scores: &mut scratch.kda_clip_scores,
                    factors: &mut scratch.kda_clip_factors,
                    row_count: GPT2_TOKEN_ROWS_U32,
                    qkv_dim: dims.qkv_dim,
                    embedding_dim: dims.embedding_dim,
                    head_count: dims.head_count,
                    head_dim: dims.head_dim,
                    tau: KDA_QK_CLIP_TAU,
                    silu_qk: (!full_attention) as u32,
                    norm_offset: (block_index as u32) * 2 * dims.head_count,
                    factor_offset: (block_index as u32) * dims.head_count,
                    precomputed_qk_norms: 1,
                })?;
        }
        Ok(())
    })?;
    Ok(())
}
