use cuda_core::DriverError;

use super::matmul::{AttentionTcMatmulContext, run_tc_matmul};
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

pub(super) fn run_dot_scores(
    ctx: &AttentionTcMatmulContext<'_>,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
) -> Result<(), DriverError> {
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
