use cuda_core::{DeviceBuffer, DriverError};

use super::matmul::{
    AttentionTcMatmulContext, run_tc_matmul_a_transposed_rhs,
    run_tc_matmul_a_transposed_rhs_sparse, run_tc_matmul_a_transposed_rhs_sparse_scaled,
    run_tc_matmul_rhs, run_tc_matmul_rhs_sparse,
};
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
        ctx.attention_window,
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
        ctx.attention_window,
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
        ctx.attention_window,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "attention gradient buffers are explicit"
)]
pub(super) fn run_grad_matmuls_sparse(
    ctx: &AttentionTcMatmulContext<'_>,
    ds: &DeviceBuffer<u16>,
    q: &DeviceBuffer<u16>,
    k: &DeviceBuffer<u16>,
    d_out: &DeviceBuffer<u16>,
    probs: &DeviceBuffer<u16>,
    tile_scales: &DeviceBuffer<f32>,
    d_q: &mut DeviceBuffer<f32>,
    d_k: &mut DeviceBuffer<f32>,
    d_v: &mut DeviceBuffer<f32>,
) -> Result<(), DriverError> {
    run_tc_matmul_rhs_sparse(
        ctx.stream,
        ctx.tc_module,
        ds,
        k,
        tile_scales,
        d_q,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
        ctx.attention_window,
    )?;
    run_tc_matmul_a_transposed_rhs_sparse(
        ctx.stream,
        ctx.tc_module,
        ds,
        q,
        tile_scales,
        d_k,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
        ctx.attention_window,
    )?;
    run_tc_matmul_a_transposed_rhs_sparse_scaled(
        ctx.stream,
        ctx.tc_module,
        probs,
        d_out,
        tile_scales,
        d_v,
        ctx.batch_head,
        ctx.seq_len,
        ctx.head_dim,
        ctx.seq_len,
        ctx.attention_window,
    )
}
