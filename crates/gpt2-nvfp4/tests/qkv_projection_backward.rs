use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionBackwardModules, AttentionQkvBackwardArgs, GPT2_N_EMBD, GPT2_QKV, HiddenState,
    HiddenVectorShape, LinearScratch, QkvVectorShape, QkvWeightShape, ResidualWeightShape,
    qkv_projection_backward,
};
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::nvfp4::{Nvfp4DecodeModule, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::transpose::TransposeModule;

mod common;
#[path = "qkv_projection_backward/data.rs"]
mod data;

use common::saved_block::{SavedBlockParts, saved_block};
use common::upload::{
    TestResult, attention_projection_tensors, upload_nvfp4_bytes, upload_zero_nvfp4,
};
use common::{assert_nonzero_finite, cuda_test_context, row_ones};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn qkv_projection_backward_runs_linear_ms_eden_path() -> TestResult {
    let (_, stream, ptx) = cuda_test_context()?;
    let transpose = TransposeModule::from_module(ptx.clone())?;
    let decode = Nvfp4DecodeModule::from_module(ptx.clone())?;
    let linear = LinearBackwardModule::from_module(ptx.clone())?;
    let quant = Nvfp4QuantModule::from_module(ptx)?;

    let qkv_input_bytes = DeviceBuffer::from_host(&stream, &data::qkv_input_bytes())?;
    let qkv_input_scales = DeviceBuffer::from_host(&stream, &data::hidden_scales())?;
    let qkv_input_globals = DeviceBuffer::from_host(&stream, &row_ones())?;
    let d_qkv = DeviceBuffer::from_host(&stream, &data::d_qkv_values())?;
    let dummy_f32 = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let dummy_u16 = DeviceBuffer::<u16>::zeroed(&stream, 1)?;

    let saved = saved_block(SavedBlockParts {
        rowwise: Nvfp4RowwiseDeviceTensor::new(
            &qkv_input_bytes,
            &qkv_input_scales,
            &qkv_input_globals,
        ),
        residual: &dummy_u16,
        mean: &dummy_f32,
        inv_std: &dummy_f32,
        qkv: &dummy_u16,
        attention_out: &dummy_u16,
        attention_log_sum_exp: &dummy_f32,
        mlp_up: &dummy_u16,
    });
    let qkv_weight = upload_nvfp4_bytes::<QkvWeightShape>(&stream, data::qkv_weight_bytes())?;
    let qkv_bias = upload_zero_nvfp4::<QkvVectorShape>(&stream)?;
    let c_proj_weight = upload_zero_nvfp4::<ResidualWeightShape>(&stream)?;
    let c_proj_bias = upload_zero_nvfp4::<HiddenVectorShape>(&stream)?;
    let projections =
        attention_projection_tensors(&qkv_weight, &qkv_bias, &c_proj_weight, &c_proj_bias);
    let mut scratch = LinearScratch::new(&stream, GPT2_N_EMBD, GPT2_QKV)?;
    let mut d_ln_1_normalized = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut d_attn_qkv_weight = DeviceBuffer::<f32>::zeroed(&stream, GPT2_N_EMBD * GPT2_QKV)?;
    let mut d_attn_qkv_bias = DeviceBuffer::<f32>::zeroed(&stream, GPT2_QKV)?;

    qkv_projection_backward(AttentionQkvBackwardArgs {
        use_full_attention: false,
        stream: &stream,
        modules: AttentionBackwardModules {
            transpose: &transpose,
            decode: &decode,
            linear: &linear,
            quant: &quant,
        },
        saved,
        projections,
        d_qkv: &d_qkv,
        d_ln_1_normalized: &mut d_ln_1_normalized,
        d_attn_qkv_weight: &mut d_attn_qkv_weight,
        d_attn_qkv_bias: &mut d_attn_qkv_bias,
        precomputed_d_qkv_amax_chunks: None,
        scratch: scratch.qkv(),
        seeds: data::seeds(),
    })?;

    assert_nonzero_finite(&d_ln_1_normalized.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_attn_qkv_weight.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_attn_qkv_bias.to_host_vec(&stream)?);
    Ok(())
}
