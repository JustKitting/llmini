use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    AttentionCoreScratchBuffers, BlockAttentionBackwardScratch, GPT2_MLP, GPT2_MLP_UP, GPT2_N_EMBD,
    GPT2_QKV, GPT2_VOCAB_SIZE, Gpt2BackwardScratch, HiddenState, LinearScratch, MlpBackwardScratch,
    QkvActivation,
};

use super::device_buffer::zero;

pub struct BackwardScratchBuffers {
    pub value_qkv_recompute: DeviceBuffer<f32>,
    pub value_first_grad: DeviceBuffer<f32>,
    value_padded_weight: DeviceBuffer<u8>,
    value_tma: rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    pub mlp_paired: DeviceBuffer<f32>,
    mlp_recompute_tma: rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors,
    final_head: LinearScratch,
    attention_c_proj: LinearScratch,
    attention_qkv: LinearScratch,
    pub attention_core: AttentionCoreScratchBuffers,
    mlp_down: LinearScratch,
    mlp_up: LinearScratch,
    // Keep these disjoint from the forward residual and QKV workspaces. Matched
    // training diagnostics show that aliasing either workspace corrupts gradients.
    d_residual_after_attention: DeviceBuffer<f32>,
    d_qkv: DeviceBuffer<f32>,
}

impl BackwardScratchBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            value_qkv_recompute: zero(
                stream,
                if gpt2_nvfp4::GPT2_VALUE_RMS {
                    QkvActivation::LEN
                } else {
                    1
                },
            )?,
            value_first_grad: zero(
                stream,
                if gpt2_nvfp4::GPT2_VALUE_RMS {
                    HiddenState::LEN
                } else {
                    1
                },
            )?,
            value_padded_weight: zero(
                stream,
                if gpt2_nvfp4::GPT2_VALUE_RMS {
                    rust_kernels_cuda::nvfp4_tma_matmul::scale_layout::sm120_scale_padded_mn_extent(
                        GPT2_QKV,
                    ) * GPT2_N_EMBD
                        / 2
                } else {
                    1
                },
            )?,
            value_tma:
                rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors::new(
                    stream,
                )?,
            mlp_paired: zero(
                stream,
                if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
                    gpt2_nvfp4::GPT2_TOKEN_ROWS * GPT2_MLP_UP
                } else {
                    1
                },
            )?,
            mlp_recompute_tma:
                rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors::new(
                    stream,
                )?,
            final_head: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_VOCAB_SIZE)?,
            attention_c_proj: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_N_EMBD)?,
            attention_qkv: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_QKV)?,
            attention_core: AttentionCoreScratchBuffers::new(stream)?,
            mlp_down: LinearScratch::new(stream, GPT2_MLP, GPT2_N_EMBD)?,
            mlp_up: LinearScratch::new(stream, GPT2_N_EMBD, GPT2_MLP_UP)?,
            d_residual_after_attention: zero(stream, HiddenState::LEN)?,
            d_qkv: zero(stream, QkvActivation::LEN)?,
        })
    }

    pub fn scratch<'a>(
        &'a mut self,
        d_hidden: &'a mut DeviceBuffer<f32>,
        d_value_residual: &'a mut DeviceBuffer<f32>,
        d_mlp_up: &'a mut DeviceBuffer<f32>,
        projection_observer: Option<&'a dyn gpt2_nvfp4::MlpProjectionObserver>,
        value_observer: Option<&'a dyn gpt2_nvfp4::ValueProjectionObserver>,
    ) -> Gpt2BackwardScratch<'a> {
        let down_linear = self.mlp_down.parts();
        let up_linear = self.mlp_up.parts();
        let (raw_first_value, d_value_residual) = if gpt2_nvfp4::GPT2_VALUE_RMS {
            (&*d_value_residual, &mut self.value_first_grad)
        } else {
            (&self.value_first_grad, d_value_residual)
        };
        Gpt2BackwardScratch {
            final_head: self.final_head.final_head(),
            attention: BlockAttentionBackwardScratch {
                value_rms: gpt2_nvfp4::ValueRmsBackwardScratch {
                    qkv: &mut self.value_qkv_recompute,
                    padded_weight: &mut self.value_padded_weight,
                    descriptors: &mut self.value_tma,
                    raw_first_value,
                    observer: value_observer,
                },
                c_proj: self.attention_c_proj.c_proj(),
                core: self.attention_core.args(),
                qkv: self.attention_qkv.qkv(),
            },
            mlp: MlpBackwardScratch {
                paired: &mut self.mlp_paired,
                tma_descriptors: &mut self.mlp_recompute_tma,
                projection_observer,
                down_linear,
                up_linear,
            },
            d_residual_after_attention: &mut self.d_residual_after_attention,
            d_hidden,
            d_qkv: &mut self.d_qkv,
            d_value_residual,
            d_mlp_up,
        }
    }
}
