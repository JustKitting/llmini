use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};
use rust_kernels_cuda::attention::AttentionModule;
use rust_kernels_cuda::canon::CanonModule;
use rust_kernels_cuda::embedding::EmbeddingModule;
use rust_kernels_cuda::f16_tc_matmul::F16TcMatmulModule;
use rust_kernels_cuda::f32_matrix_ops::F32MatrixOpsModule;
use rust_kernels_cuda::layer_norm::LayerNormModule;
use rust_kernels_cuda::layer_norm_backward::LayerNormBackwardModule;
use rust_kernels_cuda::linear_backward::LinearBackwardModule;
use rust_kernels_cuda::lm_head::LmHeadModule;
use rust_kernels_cuda::logits::LogitsModule;
use rust_kernels_cuda::loss::LossModule;
use rust_kernels_cuda::mlp::MlpModule;
use rust_kernels_cuda::next_latent::NextLatModule;
use rust_kernels_cuda::nvfp4::Nvfp4DecodeModule;
use rust_kernels_cuda::nvfp4_quant::Nvfp4QuantModule;
use rust_kernels_cuda::nvfp4_tma_matmul::{launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule};
use rust_kernels_cuda::optimizer::OptimizerModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::AppResult;
use crate::training::env::{env_nonempty, env_usize};

mod backward;

pub struct Runtime {
    cuda_module: Arc<cuda_core::CudaModule>,
    pub stream: Arc<CudaStream>,
    pub embedding: EmbeddingModule,
    pub attention: AttentionModule,
    pub value_rms: rust_kernels_cuda::value_rms::ValueRmsModule,
    pub position: rust_kernels_cuda::position::PositionModule,
    pub canon: CanonModule,
    pub f16_tc_matmul: F16TcMatmulModule,
    pub f32_ops: F32MatrixOpsModule,
    pub quant: Nvfp4QuantModule,
    pub layer_norm: LayerNormModule,
    pub mlp: MlpModule,
    pub lm_head: LmHeadModule,
    pub logits: LogitsModule,
    pub next_latent: NextLatModule,
    pub loss: LossModule,
    pub transpose: TransposeModule,
    pub decode: Nvfp4DecodeModule,
    pub tma_gemm: Nvfp4GemmModule,
    pub tma_pad: TmaMatrixPadModule,
    pub linear: LinearBackwardModule,
    pub layer_norm_backward: LayerNormBackwardModule,
    pub residual: ResidualBackwardModule,
    pub optimizer: OptimizerModule,
}

impl Runtime {
    #[cfg(test)]
    pub(in crate::training) fn value_rms_point_check_module(
        &self,
    ) -> Result<rust_kernels_cuda::value_rms::ValueRmsModule, cuda_core::DriverError> {
        rust_kernels_cuda::value_rms::ValueRmsModule::from_module(self.cuda_module.clone())
    }

    pub(in crate::training) fn kda_decay_audit_module(
        &self,
    ) -> Result<rust_kernels_cuda::kda_decay_audit::DecayAuditModule, cuda_core::DriverError> {
        rust_kernels_cuda::kda_decay_audit::DecayAuditModule::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn mhc_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc::kernels::LoadedModule, cuda_core::DriverError> {
        rust_kernels_cuda::mhc::kernels::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn mhc_coefficients_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc_coefficients::MhcCoefficientsModule, cuda_core::DriverError>
    {
        rust_kernels_cuda::mhc_coefficients::MhcCoefficientsModule::from_module(
            self.cuda_module.clone(),
        )
    }
    pub(in crate::training) fn mhc_reverse_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc_reverse::MhcReverseModule, cuda_core::DriverError> {
        rust_kernels_cuda::mhc_reverse::MhcReverseModule::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn mhc_post_reuse_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc_post_reuse::MhcPostReuseModule, cuda_core::DriverError> {
        rust_kernels_cuda::mhc_post_reuse::MhcPostReuseModule::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn mhc_rms_quant_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc_rms_quant::MhcRmsQuantModule, cuda_core::DriverError> {
        rust_kernels_cuda::mhc_rms_quant::MhcRmsQuantModule::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn mhc_window_post_module(
        &self,
    ) -> Result<rust_kernels_cuda::mhc_window_post::MhcWindowPostModule, cuda_core::DriverError>
    {
        rust_kernels_cuda::mhc_window_post::MhcWindowPostModule::from_module(
            self.cuda_module.clone(),
        )
    }
    pub fn new() -> AppResult<Self> {
        if gpt2_nvfp4::GPT2_STATIONARY_ROPE
            && crate::training::env::env_bool("TRAIN_PARTIAL_KEY_OFFSET") == Some(false)
        {
            return Err("stationary RoPE requires its stationary-half key offset; unset TRAIN_PARTIAL_KEY_OFFSET=0".into());
        }
        let ctx = CudaContext::new(gpu_device_index())?;
        let stream = ctx.new_stream()?;
        let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
        let runtime = Self {
            cuda_module: ptx.clone(),
            stream,
            embedding: EmbeddingModule::from_module(ptx.clone())?,
            attention: AttentionModule::from_module(ptx.clone())?,
            value_rms: rust_kernels_cuda::value_rms::ValueRmsModule::from_module(ptx.clone())?,
            position: rust_kernels_cuda::position::PositionModule::from_module(ptx.clone())?,
            canon: CanonModule::from_module(ptx.clone())?,
            f16_tc_matmul: F16TcMatmulModule::from_module(ptx.clone())?,
            f32_ops: F32MatrixOpsModule::from_module(ptx.clone())?,
            quant: Nvfp4QuantModule::from_module(ptx.clone())?,
            layer_norm: LayerNormModule::from_module(ptx.clone())?,
            mlp: MlpModule::from_module(ptx.clone())?,
            lm_head: LmHeadModule::from_module(ptx.clone())?,
            logits: LogitsModule::from_module(ptx.clone())?,
            next_latent: NextLatModule::from_module(ptx.clone())?,
            loss: LossModule::from_module(ptx.clone())?,
            transpose: TransposeModule::from_module(ptx.clone())?,
            decode: Nvfp4DecodeModule::from_module(ptx.clone())?,
            tma_gemm: Nvfp4GemmModule::from_module(ptx.clone())?,
            tma_pad: TmaMatrixPadModule::from_module(ptx.clone())?,
            linear: LinearBackwardModule::from_module(ptx.clone())?,
            layer_norm_backward: LayerNormBackwardModule::from_module(ptx.clone())?,
            residual: ResidualBackwardModule::from_module(ptx.clone())?,
            optimizer: OptimizerModule::from_module(ptx)?,
        };
        runtime.mlp.verify_graph(runtime.stream.as_ref())?;
        runtime.value_rms.verify_graph(runtime.stream.as_ref())?;
        runtime.position.verify_graph(runtime.stream.as_ref())?;
        runtime
            .optimizer
            .verify_storage_layout(runtime.stream.as_ref())?;
        Ok(runtime)
    }
}

impl Runtime {
    pub(in crate::training) fn engram_multi_module(
        &self,
    ) -> Result<rust_kernels_cuda::engram_multi::kernels::LoadedModule, cuda_core::DriverError>
    {
        rust_kernels_cuda::engram_multi::kernels::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn engram_module(
        &self,
    ) -> Result<rust_kernels_cuda::engram::kernels::LoadedModule, cuda_core::DriverError> {
        rust_kernels_cuda::engram::kernels::from_module(self.cuda_module.clone())
    }
    pub(in crate::training) fn engram_four_gram_hash_module(
        &self,
    ) -> Result<rust_kernels_cuda::engram::four_gram_hash::LoadedModule, cuda_core::DriverError>
    {
        rust_kernels_cuda::engram::four_gram_hash::from_module(self.cuda_module.clone())
    }
}

impl Runtime {
    pub(in crate::training) fn latent_feedback_module(
        &self,
    ) -> Result<rust_kernels_cuda::latent_feedback::kernels::LoadedModule, cuda_core::DriverError>
    {
        rust_kernels_cuda::latent_feedback::kernels::from_module(self.cuda_module.clone())
    }
}

fn gpu_device_index() -> usize {
    env_usize("CUDA_DEVICE_INDEX").unwrap_or(0)
}

pub(in crate::training) fn ptx_path() -> String {
    env_nonempty("CUDA_OXIDE_PTX_PATH")
        .unwrap_or_else(|| format!("{}/rust_kernels_cuda.ptx", env!("CARGO_MANIFEST_DIR")))
}
