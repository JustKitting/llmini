use cuda_core::{DeviceBuffer, DriverError};

use super::matmul::{AttentionTcMatmulContext, run_tc_matmul_a_transposed_rhs, run_tc_matmul_rhs};
use super::types::CausalAttentionBackwardTcScratch;

pub(super) fn run_grad_matmuls(
    ctx: &AttentionTcMatmulContext<'_>,
    scratch: &mut CausalAttentionBackwardTcScratch<'_>,
    forward_probs_f16: Option<&DeviceBuffer<u16>>,
) -> Result<(), DriverError> {
    run_tc_matmul_rhs(
        ctx.stream,
        ctx.tc_module,
        scratch.ds_half,
        scratch.k,
        scratch.d_q,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    run_tc_matmul_a_transposed_rhs(
        ctx.stream,
        ctx.tc_module,
        scratch.ds_half,
        scratch.q,
        scratch.d_k,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )?;
    let probs_half = forward_probs_f16.unwrap_or(&*scratch.p_half);
    run_tc_matmul_a_transposed_rhs(
        ctx.stream,
        ctx.tc_module,
        probs_half,
        scratch.d_out,
        scratch.d_v,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
    )
}
