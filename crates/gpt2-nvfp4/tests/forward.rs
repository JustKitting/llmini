use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    GPT2_BATCH_SIZE, GPT2_SEQ_LEN, GPT2_TOKEN_ROWS, Gpt2, Gpt2ForwardArgs, TokenEmbeddingArgs,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;

mod common;
#[path = "forward/scratch.rs"]
mod scratch;

use common::cuda_test_context;
use common::upload::{TestResult, block::upload_block, upload_layer_norm, upload_nvfp4};
use scratch::ForwardScratch;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn gpt2_forward_runs_through_tied_lm_head() -> TestResult {
    let (_, stream, module) = cuda_test_context()?;
    let embedding_module = EmbeddingModule::from_module(module.clone())?;
    let attention_module = AttentionModule::from_module(module.clone())?;
    let attention_tc_module = F16TcMatmulModule::from_module(module.clone())?;
    let quant_module = Nvfp4QuantModule::from_module(module.clone())?;
    let layer_norm_module = LayerNormModule::from_module(module.clone())?;
    let mlp_module = MlpModule::from_module(module.clone())?;
    let lm_head_module = LmHeadModule::from_module(module)?;

    let mut model = Gpt2::new();
    model.init(0x4750_5432);
    let weights = model
        .weights()
        .expect("Gpt2::init must create model weights");

    let token_embedding = upload_nvfp4(&stream, &weights.embeddings.wte)?;
    let blocks = weights
        .h
        .iter()
        .map(|block| upload_block(&stream, block))
        .collect::<TestResult<Vec<_>>>()?;
    let ln_f = upload_layer_norm(&stream, &weights.ln_f)?;

    let tokens = token_ids();
    let tokens_dev = DeviceBuffer::from_host(&stream, &tokens)?;
    let mut scratch = ForwardScratch::new(&stream)?;

    model.forward(Gpt2ForwardArgs {
        input: gpt2_nvfp4::Gpt2Input::Tokens(TokenEmbeddingArgs {
            module: &embedding_module,
            stream: &stream,
            tokens: &tokens_dev,
            token_embedding: token_embedding.device(),
            batch_size: GPT2_BATCH_SIZE as u32,
            seq_len: GPT2_SEQ_LEN as u32,
            row_count: GPT2_TOKEN_ROWS as u32,
            residual: &mut scratch.residual,
            normalized: &mut scratch.normalized,
            normalized_amax: &mut scratch.normalized_amax,
            mean: &mut scratch.mean,
            inv_std: &mut scratch.inv_std,
        }),
        attention_module: &attention_module,
        attention_tc_module: &attention_tc_module,
        quant_module: &quant_module,
        layer_norm_module: &layer_norm_module,
        mlp_module: &mlp_module,
        lm_head_module: &lm_head_module,
        hidden_nvfp4: scratch.hidden_nvfp4.scratch(),
        attention_tc_scratch: scratch.attention_tc.args(),
        mlp_activation_nvfp4: scratch.mlp_activation_nvfp4.scratch(),
        attention: std::array::from_fn(|i| blocks[i].attention_tensors()),
        block_ln_1: std::array::from_fn(|i| blocks[i].ln_1.tensors()),
        block_ln_2: std::array::from_fn(|i| blocks[i].ln_2.tensors()),
        mlp: std::array::from_fn(|i| blocks[i].mlp_tensors()),
        ln_f: ln_f.tensors(),
        attention_qkv: &mut scratch.qkv,
        attention_log_sum_exp: &mut scratch.attention_log_sum_exp,
        mlp_pre_activation: &mut scratch.mlp_pre_activation,
        mlp_activation: &mut scratch.mlp_activation,
        logits: &mut scratch.logits,
        tape: None,
    })?;

    let logits = scratch.logits.to_host_vec(&stream)?;
    common::assert_nonzero_finite(&logits);
    Ok(())
}

fn token_ids() -> Vec<u32> {
    (0..GPT2_TOKEN_ROWS).map(|i| (i % 127) as u32).collect()
}
