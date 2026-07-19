use cuda_core::DeviceBuffer;
use gpt2_nvfp4::{
    AttentionBackwardModules, BlockAttentionBackwardArgs, BlockAttentionBackwardModules,
    BlockAttentionBackwardSeeds, GPT2_TOKEN_ROWS, GPT2_VALUE_RESIDUAL_START_LAYER,
    GPT2_XSA_ALPHA_STORAGE, Gpt2Rng, HiddenState, attention_side_backward,
};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;
use rust_kernels_cuda::transpose::TransposeModule;

#[path = "block_attention_backward/buffers/mod.rs"]
mod buffers;
mod common;
#[path = "block_attention_backward/data.rs"]
mod data;
#[path = "block_attention_backward/scratch.rs"]
mod scratch;

use common::upload::TestResult;
use common::{assert_nonzero_finite, cuda_test_context};

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn block_attention_side_backward_runs_full_chain() -> TestResult {
    let (_, stream, ptx) = cuda_test_context()?;
    let saved = buffers::SavedBuffers::new(&stream)?;
    let weights = buffers::WeightBuffers::new(&stream)?;
    let mut grads = buffers::GradBuffers::new(&stream)?;
    let mut scratch = scratch::BlockAttentionScratch::new(&stream)?;
    let mut rng = Gpt2Rng::new(0x4154_544e);
    let mut d_residual_in_chunk_amax = DeviceBuffer::<f32>::zeroed(&stream, GPT2_TOKEN_ROWS)?;
    let mut d_value_residual = DeviceBuffer::<f32>::zeroed(&stream, HiddenState::LEN)?;
    let mut d_xsa_alphas = DeviceBuffer::<f32>::zeroed(&stream, GPT2_XSA_ALPHA_STORAGE)?;

    let (d_residual_after_attention, d_residual_in, d_hidden, d_qkv, backward_grads) =
        grads.block();
    attention_side_backward(BlockAttentionBackwardArgs {
        block_index: GPT2_VALUE_RESIDUAL_START_LAYER,
        use_full_attention: false,
        reuse_forward_probs: false,
        stream: &stream,
        modules: BlockAttentionBackwardModules {
            residual: &ResidualBackwardModule::from_module(ptx.clone())?,
            layer_norm: &LayerNormBackwardModule::from_module(ptx.clone())?,
            attention: &AttentionModule::from_module(ptx.clone())?,
            f16_tc: &F16TcMatmulModule::from_module(ptx.clone())?,
            linear: AttentionBackwardModules {
                transpose: &TransposeModule::from_module(ptx.clone())?,
                decode: &Nvfp4DecodeModule::from_module(ptx.clone())?,
                linear: &LinearBackwardModule::from_module(ptx.clone())?,
                quant: &Nvfp4QuantModule::from_module(ptx)?,
            },
        },
        saved: saved.block(),
        ln_1: weights.ln_1(),
        projections: weights.projections(),
        d_residual_after_attention,
        precomputed_d_residual_after_attention_amax_chunks: None,
        d_residual_in,
        d_residual_in_chunk_amax: &mut d_residual_in_chunk_amax,
        d_hidden,
        d_qkv,
        d_value_residual: &mut d_value_residual,
        d_xsa_alphas: &mut d_xsa_alphas,
        grads: backward_grads,
        scratch: scratch.block(),
        seeds: BlockAttentionBackwardSeeds::from_rng(&mut rng),
    })?;

    assert_nonzero_finite(&grads.d_residual_in.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_hidden.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_qkv.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_value_residual.to_host_vec(&stream)?);
    assert_nonzero_finite(&d_xsa_alphas.to_host_vec(&stream)?[..32]);
    assert_nonzero_finite(&grads.d_attn_qkv_weight.to_host_vec(&stream)?);
    assert_nonzero_finite(&grads.d_attn_c_proj_weight.to_host_vec(&stream)?);
    Ok(())
}
