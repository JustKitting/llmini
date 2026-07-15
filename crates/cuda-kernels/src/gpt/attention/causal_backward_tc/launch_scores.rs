use cuda_core::DriverError;

use super::matmul::{AttentionTcMatmulContext, run_tc_matmul, run_tc_matmul_lower_ds};
use super::types::CausalAttentionBackwardTcScratch;

pub(super) fn run_pair_scores(
    ctx: &AttentionTcMatmulContext<'_>,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
) -> Result<(), DriverError> {
    run_tc_matmul(
        ctx.stream,
        ctx.tc_module,
        scratch.q,
        scratch.k,
        scratch.scores,
        ctx.batch_head,
        ctx.seq_len,
        ctx.seq_len,
        ctx.head_dim,
    )?;
    run_tc_matmul(
        ctx.stream,
        ctx.tc_module,
        scratch.d_out,
        scratch.v,
        scratch.dot,
        ctx.batch_head,
        ctx.seq_len,
        ctx.seq_len,
        ctx.head_dim,
    )
}

pub(super) fn run_ds_scores(
    ctx: &AttentionTcMatmulContext<'_>,
    d_out: &cuda_core::DeviceBuffer<u16>,
    v: &cuda_core::DeviceBuffer<u16>,
    probs: &cuda_core::DeviceBuffer<u16>,
    softmax_d: &cuda_core::DeviceBuffer<f32>,
    ds: &mut cuda_core::DeviceBuffer<u16>,
) -> Result<(), DriverError> {
    run_tc_matmul_lower_ds(
        ctx.stream,
        ctx.tc_module,
        d_out,
        v,
        probs,
        softmax_d,
        ds,
        ctx.batch_head,
        ctx.seq_len,
        ctx.seq_len,
        ctx.head_dim,
    )
}
