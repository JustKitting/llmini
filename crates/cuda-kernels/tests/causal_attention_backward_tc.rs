use std::error::Error;

use cuda_core::DeviceBuffer;
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardTcArgs};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;

#[path = "causal_attention_backward_tc/case.rs"]
mod case;
mod common;
#[path = "causal_attention_backward_tc/f16.rs"]
mod f16;
#[path = "causal_attention_backward_tc/reference.rs"]
mod reference;
#[path = "causal_attention_backward_tc/scratch.rs"]
mod scratch;
#[path = "causal_attention_backward_tc/shape.rs"]
mod shape;

use scratch::TcScratchBuffers;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn materialized_tc_backward_matches_reference() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = common::cuda_test_context()?;
    let attention = AttentionModule::from_module(ptx.clone())?;
    let tc = F16TcMatmulModule::from_module(ptx)?;
    let case = case::simple_case();

    let (qkv, qkv_ref) = f16::saved_f16(&stream, &tc, &case.qkv)?;
    let (out, out_ref) = f16::saved_f16(&stream, &tc, &case.out)?;
    let (_, d_out_ref) = f16::saved_f16(&stream, &tc, &case.d_out)?;
    let d_out = DeviceBuffer::from_host(&stream, &case.d_out)?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &case.log_sum_exp)?;
    let expected = reference::backward(
        &qkv_ref,
        &out_ref,
        &case.d_out,
        &d_out_ref,
        &case.log_sum_exp,
    );
    let mut tc_softmax_d = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut tc_qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * shape::HEADS)?;
    let mut tc_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut tc_grad_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut scratch = TcScratchBuffers::new(&stream)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: false,
        forward_probs_f16: None,
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut tc_softmax_d,
        qk_norm_max: &mut tc_qk_norm_max,
        d_qkv: &mut tc_grad,
        d_qkv_chunk_amax: &mut tc_grad_chunk_amax,
        scratch: scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
        qk_norm_offset: 0,
    })?;

    let recomputed = tc_grad.to_host_vec(&stream)?;
    common::assert_slice_close(&recomputed, &expected, 1.0e-6);

    let mut saved_probs = vec![0_u16; shape::HEADS * shape::TOKEN_COUNT * shape::TOKEN_COUNT];
    for head in 0..shape::HEADS {
        let base = head * shape::TOKEN_COUNT * shape::TOKEN_COUNT;
        saved_probs[base] = 0x3c00;
        saved_probs[base + shape::TOKEN_COUNT] = 0x3800;
        saved_probs[base + shape::TOKEN_COUNT + 1] = 0x3800;
    }
    let mut reuse_softmax_d =
        DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::HEADS)?;
    let mut reuse_qk_norm_max = DeviceBuffer::<f32>::zeroed(&stream, 2 * shape::HEADS)?;
    let mut reuse_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut reuse_grad_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut reuse_scratch = TcScratchBuffers::new(&stream)?;
    let saved_probs = DeviceBuffer::from_host(&stream, &saved_probs)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        reuse_forward_probs: true,
        forward_probs_f16: Some(&saved_probs),
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        kda_v_new: None,
        kda_akk_inv: None,
        kda_w: None,
        kda_aqk: None,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut reuse_softmax_d,
        qk_norm_max: &mut reuse_qk_norm_max,
        d_qkv: &mut reuse_grad,
        d_qkv_chunk_amax: &mut reuse_grad_chunk_amax,
        scratch: reuse_scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
        qk_norm_offset: 0,
    })?;
    common::assert_slice_close(&reuse_grad.to_host_vec(&stream)?, &recomputed, 1.0e-6);
    Ok(())
}
