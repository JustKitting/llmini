use rust_kernels_cuda::nvfp4_tma_matmul::launcher::Nvfp4GemmModule;
mod hidden;
mod output;

pub(super) use hidden::{projection_gelu1, projection_gelu2};
pub(super) use output::output_and_loss;

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tma_matmul::tma::TmaNvfp4DeviceScaleDescriptors;

#[expect(
    clippy::too_many_arguments,
    reason = "TMA projection uses explicit tensor and scratch buffers"
)]
fn project_affine_tma(
    stream: &CudaStream,
    tma: &Nvfp4GemmModule,
    descriptors: &mut TmaNvfp4DeviceScaleDescriptors,
    input: Nvfp4RowwiseDeviceTensor<'_>,
    weight: Nvfp4DeviceTensor<'_>,
    bias: Nvfp4DeviceTensor<'_>,
    out: &mut DeviceBuffer<f32>,
    token_count: u32,
    input_dim: u32,
    output_dim: u32,
) -> Result<(), DriverError> {
    tma.prepare_tma_nvfp4_csf_weight_into(
        stream,
        input.bytes,
        input.scales,
        weight.bytes,
        weight.scales,
        token_count,
        input_dim,
        output_dim,
        descriptors,
    )?;
    tma.gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
        stream,
        descriptors,
        out,
        bias,
        token_count,
        input_dim,
        output_dim,
        output_dim,
        input.global_scales,
        weight.global_scale,
    )
}
