use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::forward;
use super::tensors::MlpForwardArgs;
use crate::random::InitRng;
use crate::types::{HiddenStateDevice, MlpDownLinear, MlpUpLinear};

#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub c_fc: MlpUpLinear,
    pub c_proj: MlpDownLinear,
}

impl MlpWeights {
    pub const fn initializer_name() -> &'static str {
        if crate::config::GPT2_MLP_REFERENCE_NORMAL_INIT {
            "reference-normal-nvfp4"
        } else {
            "legacy-packed-smooth"
        }
    }

    /// Shared forward/recompute producer. Both operands remain native NVFP4;
    /// FP32 here is the ordinary accumulator/activation workspace.
    pub fn project_paired(
        tma: &Nvfp4GemmModule,
        stream: &CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'_>,
        up: super::tensors::MlpUpTensors<'_>,
        descriptors: &mut TmaNvfp4DeviceScaleDescriptors,
        paired: &mut DeviceBuffer<f32>,
        n: u32,
    ) -> Result<(), DriverError> {
        let k = crate::GPT2_EMBEDDING_DIM;
        let output = crate::GPT2_MLP_UP_DIM;
        if paired.len() < (n * output) as usize {
            return Err(DriverError(1));
        }
        tma.prepare_tma_nvfp4_csf_weight_into(
            stream,
            input.bytes,
            input.scales,
            up.weight_device.bytes,
            up.weight_device.scales,
            n,
            k,
            output,
            descriptors,
        )?;
        tma.gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
            stream,
            descriptors,
            paired,
            up.bias,
            n,
            k,
            output,
            output,
            input.global_scales,
            up.weight_device.global_scale,
        )
    }
    pub fn forward_branch<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        let routed = crate::uses_block_topk_mlp(args.block_index);
        Self::forward_with_routing(args, false, routed)
    }

    pub(crate) fn forward_with_routing<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
        add_residual: bool,
        routed: bool,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward_impl(args, add_residual, routed)
    }
    pub(crate) fn init(rng: &mut InitRng, residual_projection_scale: f32) -> Self {
        if crate::config::GPT2_MLP_REFERENCE_NORMAL_INIT {
            return Self {
                c_fc: MlpUpLinear::init_reference_normal(rng, 0.02),
                c_proj: MlpDownLinear::init_reference_normal(rng, residual_projection_scale),
            };
        }
        Self {
            c_fc: MlpUpLinear::init(rng),
            c_proj: MlpDownLinear::init_with_weight_scale(rng, residual_projection_scale),
        }
    }

    pub fn forward<'a, 'scratch>(
        args: MlpForwardArgs<'a, 'scratch>,
    ) -> Result<HiddenStateDevice<'a>, DriverError> {
        forward::forward(args)
    }
}
