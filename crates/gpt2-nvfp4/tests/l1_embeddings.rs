use std::error::Error;

use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    GPT2_CONTEXT_LEN, GPT2_N_EMBD, GPT2_TOKEN_ROWS, HiddenState, Nvfp4Shape, TokenEmbeddingShape,
};
use rust_kernels_cuda::embedding::{EmbeddingArgs, EmbeddingModule};
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

mod common;

use common::cuda_test_context;
use common::nvfp4::{E4M3_ONE, set_e2m1_one};

const TOLERANCE: f32 = 1.0e-7;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn embedding_forward_decodes_token_embeddings_to_residual_only() -> Result<(), Box<dyn Error>> {
    let (_, stream, ptx) = cuda_test_context()?;
    let module = EmbeddingModule::from_module(ptx)?;

    let mut tokens = vec![0_u32; GPT2_TOKEN_ROWS];
    tokens[0] = 7;
    tokens[1] = 11;

    let mut token_embedding_bytes = vec![0_u8; TokenEmbeddingShape::BYTE_LEN];
    set_e2m1_one(&mut token_embedding_bytes, 7 * GPT2_N_EMBD);
    set_e2m1_one(&mut token_embedding_bytes, 7 * GPT2_N_EMBD + 37);
    set_e2m1_one(&mut token_embedding_bytes, 7 * GPT2_N_EMBD + 1900);
    set_e2m1_one(&mut token_embedding_bytes, 11 * GPT2_N_EMBD + 2);

    let token_embedding_scales = vec![E4M3_ONE; TokenEmbeddingShape::SCALE_LEN];
    let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
    let token_embedding_bytes_dev = DeviceBuffer::from_host(&stream, &token_embedding_bytes)?;
    let token_embedding_scales_dev = DeviceBuffer::from_host(&stream, &token_embedding_scales)?;
    let token_embedding_global_scale_dev = DeviceBuffer::from_host(&stream, &[1.0_f32])?;

    let mut residual_dev = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let normalized_sentinel = vec![123.0_f32; HiddenState::LEN];
    let amax_sentinel = vec![456.0_f32; GPT2_CONTEXT_LEN];
    let normalized_dev = DeviceBuffer::from_host(&stream, &normalized_sentinel)?;
    let amax_dev = DeviceBuffer::from_host(&stream, &amax_sentinel)?;

    module.token_embedding(EmbeddingArgs {
        stream: &stream,
        tokens: &tokens_dev,
        token_embedding: Nvfp4DeviceTensor::new(
            &token_embedding_bytes_dev,
            &token_embedding_scales_dev,
            &token_embedding_global_scale_dev,
        ),
        residual: &mut residual_dev,
        hidden_len: HiddenState::LEN as u32,
        embedding_dim: GPT2_N_EMBD as u32,
    })?;

    let residual = residual_dev.to_host_vec(&stream)?;
    let normalized = normalized_dev.to_host_vec(&stream)?;
    let amax = amax_dev.to_host_vec(&stream)?;

    assert_value(residual[0], 1.0);
    assert_value(residual[37], 1.0);
    assert_value(residual[1900], 1.0);
    assert_value(residual[2], 0.0);
    assert_value(residual[GPT2_N_EMBD + 2], 1.0);
    assert_value(residual[GPT2_N_EMBD], 0.0);
    assert!(normalized.iter().all(|value| *value == 123.0));
    assert!(amax.iter().all(|value| *value == 456.0));
    Ok(())
}

fn assert_value(actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= TOLERANCE,
        "actual={actual:.8e} expected={expected:.8e} error={error:.8e}"
    );
}
