use std::error::Error;

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionCoreBackwardArgs, AttentionCoreScratchBuffers, GPT2_BATCH_SIZE, GPT2_N_EMBD,
    GPT2_N_HEAD, GPT2_QKV, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, HiddenState, QkvActivation,
    causal_attention_backward as gpt2_causal_attention_backward,
};
use rust_kernels_cuda::attention::{AttentionModule, CausalAttentionBackwardTcArgs};
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

mod common;
#[path = "attention_core_backward/data.rs"]
mod data;

use common::saved_block::{SavedBlockParts, saved_block};
use common::{assert_nonzero_finite, attention_log_sum_exp_values, cuda_test_context, float_bits};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn causal_attention_backward_wrapper_matches_direct_kernel() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = cuda_test_context()?;
    let module = AttentionModule::from_module(ptx.clone())?;
    let tc_module = F16TcMatmulModule::from_module(ptx)?;

    let qkv = DeviceBuffer::from_host(&stream, &vec![0x3c00_u16; QkvActivation::LEN])?;
    let attention_out = DeviceBuffer::from_host(&stream, &vec![0x2e66_u16; HiddenState::LEN])?;
    let d_out = DeviceBuffer::from_host(&stream, &data::d_out_values())?;
    let log_sum_exp = DeviceBuffer::from_host(&stream, &attention_log_sum_exp_values())?;
    let dummy = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let dummy_u16 = DeviceBuffer::<u16>::zeroed(&stream, 1)?;
    let dummy_bytes = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_scales = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    let dummy_global_scales = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let saved = saved_block(SavedBlockParts {
        rowwise: Nvfp4RowwiseDeviceTensor::new(&dummy_bytes, &dummy_scales, &dummy_global_scales),
        residual: &dummy_u16,
        mean: &dummy,
        inv_std: &dummy,
        qkv: &qkv,
        attention_out: &attention_out,
        attention_log_sum_exp: &log_sum_exp,
        mlp_up: &dummy_u16,
    });
    let mut wrapper_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut direct_d_qkv = DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN)?;
    let mut wrapper_d_qkv_chunk_amax =
        DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN.div_ceil(32))?;
    let mut direct_d_qkv_chunk_amax =
        DeviceBuffer::<f32>::zeroed(&stream, QkvActivation::LEN.div_ceil(32))?;
    let mut wrapper_scratch = AttentionCoreScratchBuffers::new(&stream)?;
    let mut direct_scratch = AttentionCoreScratchBuffers::new(&stream)?;

    let wrapper_chunk_count = gpt2_causal_attention_backward(AttentionCoreBackwardArgs {
        block_index: 0,
        use_full_attention: false,
        reuse_forward_probs: false,
        stream: &stream,
        module: &module,
        tc_module: &tc_module,
        saved,
        d_attention_out: &d_out,
        d_qkv: &mut wrapper_d_qkv,
        d_qkv_chunk_amax: &mut wrapper_d_qkv_chunk_amax,
        scratch: wrapper_scratch.args(),
    })?
    .expect("KDA backward must return producer amax chunks");

    let direct_core = direct_scratch.args();
    let direct_chunk_count =
        module.kda_attention_backward_tc_detached_chunk_state(CausalAttentionBackwardTcArgs {
            reuse_forward_probs: false,
            forward_probs_f16: None,
            stream: &stream,
            tc_module: &tc_module,
            qkv: &qkv,
            attention_out: &attention_out,
            kda_v_new: None,
            kda_akk_inv: None,
            kda_w: None,
            kda_aqk: None,
            d_out: &d_out,
            log_sum_exp: &log_sum_exp,
            softmax_d: direct_core.softmax_d,
            qk_norm_max: direct_core.qk_norm_max,
            d_qkv: &mut direct_d_qkv,
            d_qkv_chunk_amax: &mut direct_d_qkv_chunk_amax,
            scratch: direct_core.tc,
            row_count: GPT2_TOKEN_ROWS as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            batch_size: GPT2_BATCH_SIZE as u32,
            embedding_dim: GPT2_N_EMBD as u32,
            qkv_dim: GPT2_QKV as u32,
            head_count: GPT2_N_HEAD as u32,
            head_dim: (GPT2_N_EMBD / GPT2_N_HEAD) as u32,
            attention_window: GPT2_SEQ_LEN as u32,
            qk_norm_offset: 0,
        })?;

    let wrapper = wrapper_d_qkv.to_host_vec(&stream)?;
    let direct = direct_d_qkv.to_host_vec(&stream)?;
    let wrapper_chunks = wrapper_d_qkv_chunk_amax.to_host_vec(&stream)?;
    let direct_chunks = direct_d_qkv_chunk_amax.to_host_vec(&stream)?;
    assert_nonzero_finite(&wrapper);
    assert_eq!(float_bits(&wrapper), float_bits(&direct));
    assert_eq!(wrapper_chunk_count, direct_chunk_count);
    assert_eq!(
        float_bits(&wrapper_chunks[..wrapper_chunk_count as usize]),
        float_bits(&direct_chunks[..direct_chunk_count as usize])
    );
    let output_amax = wrapper.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let chunk_amax = wrapper_chunks[..wrapper_chunk_count as usize]
        .iter()
        .copied()
        .fold(0.0, f32::max);
    assert_eq!(output_amax.to_bits(), chunk_amax.to_bits());

    Ok(())
}
