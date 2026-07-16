mod hidden;
mod output;

pub(super) use hidden::{projection_gelu1, projection_gelu2};
pub(super) use output::output_and_loss;

use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule, scale_pack::Sm120ScalePackModule,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

#[expect(
    clippy::too_many_arguments,
    reason = "TMA projection uses explicit tensor and scratch buffers"
)]
fn project_affine_tma(
    stream: &CudaStream,
    tma: &Nvfp4GemmModule,
    scale_pack: &Sm120ScalePackModule,
    descriptors: &mut TmaNvfp4DeviceScaleDescriptors,
    input_scale_packed: &mut DeviceBuffer<u8>,
    weight_scale_packed: &mut DeviceBuffer<u8>,
    input: Nvfp4RowwiseDeviceTensor<'_>,
    weight: Nvfp4DeviceTensor<'_>,
    bias: Nvfp4DeviceTensor<'_>,
    out: &mut DeviceBuffer<f32>,
    token_count: u32,
    input_dim: u32,
    output_dim: u32,
) -> Result<(), DriverError> {
    scale_pack.pack(
        stream,
        input.scales,
        input_scale_packed,
        token_count,
        input_dim,
    )?;
    scale_pack.pack(
        stream,
        weight.scales,
        weight_scale_packed,
        output_dim,
        input_dim,
    )?;
    tma.prepare_tma_nvfp4_device_scales_into(
        stream,
        input.bytes,
        input_scale_packed,
        weight.bytes,
        weight_scale_packed,
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
