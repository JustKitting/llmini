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
    let mut tc_grad = DeviceBuffer::<f32>::zeroed(&stream, shape::TOKEN_COUNT * shape::QKV_DIM)?;
    let mut scratch = TcScratchBuffers::new(&stream)?;
    attention.causal_attention_backward_tc(CausalAttentionBackwardTcArgs {
        stream: &stream,
        tc_module: &tc,
        qkv: &qkv,
        attention_out: &out,
        kda_v_new: None,
        d_out: &d_out,
        log_sum_exp: &log_sum_exp,
        softmax_d: &mut tc_softmax_d,
        d_qkv: &mut tc_grad,
        scratch: scratch.args(),
        row_count: shape::TOKEN_COUNT as u32,
        seq_len: shape::TOKEN_COUNT as u32,
        batch_size: 1,
        embedding_dim: shape::EMBEDDING as u32,
        qkv_dim: shape::QKV_DIM as u32,
        head_count: shape::HEADS as u32,
        head_dim: shape::HEAD_DIM as u32,
    })?;

    common::assert_slice_close(&tc_grad.to_host_vec(&stream)?, &expected, 1.0e-6);
    Ok(())
}
