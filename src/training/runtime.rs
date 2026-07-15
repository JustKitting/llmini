use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream};
use rust_kernels_cuda::attention::AttentionModule;
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
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, pad::TmaMatrixPadModule, scale_pack::Sm120ScalePackModule,
};
use rust_kernels_cuda::optimizer::OptimizerModule;
use rust_kernels_cuda::residual::ResidualBackwardModule;
use rust_kernels_cuda::transpose::TransposeModule;

use crate::AppResult;
use crate::training::env::env_usize;

mod backward;

pub struct Runtime {
    pub stream: Arc<CudaStream>,
    pub embedding: EmbeddingModule,
    pub attention: AttentionModule,
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
    pub tma_scale_pack: Sm120ScalePackModule,
    pub tma_pad: TmaMatrixPadModule,
    pub linear: LinearBackwardModule,
    pub layer_norm_backward: LayerNormBackwardModule,
    pub residual: ResidualBackwardModule,
    pub optimizer: OptimizerModule,
}

impl Runtime {
    pub fn new() -> AppResult<Self> {
        let ctx = CudaContext::new(gpu_device_index())?;
        let stream = ctx.new_stream()?;
        let ptx = ctx.load_module_from_file(ptx_path().as_str())?;
        Ok(Self {
            stream,
            embedding: EmbeddingModule::from_module(ptx.clone())?,
            attention: AttentionModule::from_module(ptx.clone())?,
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
            tma_scale_pack: Sm120ScalePackModule::from_module(ptx.clone())?,
            tma_pad: TmaMatrixPadModule::from_module(ptx.clone())?,
            linear: LinearBackwardModule::from_module(ptx.clone())?,
            layer_norm_backward: LayerNormBackwardModule::from_module(ptx.clone())?,
            residual: ResidualBackwardModule::from_module(ptx.clone())?,
            optimizer: OptimizerModule::from_module(ptx)?,
        })
    }
}

fn gpu_device_index() -> usize {
    env_usize("CUDA_DEVICE_INDEX").unwrap_or(0)
}

fn ptx_path() -> String {
    format!("{}/rust_kernels_cuda.ptx", env!("CARGO_MANIFEST_DIR"))
}
