//! The identical NVFP4 affine producer used by forward and value-RMS replay.
use super::AttentionProjectionTensors;
use crate::AttentionDims;
use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_tma_matmul::{
    launcher::Nvfp4GemmModule,
    pad::{TmaMatrixPadModule, U4RowPadArgs},
    scale_layout::sm120_scale_padded_mn_extent,
    tma::TmaNvfp4DeviceScaleDescriptors,
};

pub fn project_attention_qkv(
    stream: &CudaStream,
    tma: &Nvfp4GemmModule,
    pad: &TmaMatrixPadModule,
    descriptors: &mut TmaNvfp4DeviceScaleDescriptors,
    padded_weight: &mut DeviceBuffer<u8>,
    input: Nvfp4RowwiseDeviceTensor<'_>,
    projections: AttentionProjectionTensors<'_>,
    out: &mut DeviceBuffer<f32>,
    rows: u32,
    dims: AttentionDims,
) -> Result<(), DriverError> {
    let padded = sm120_scale_padded_mn_extent(dims.qkv_dim as usize) as u32;
    if padded != dims.qkv_dim {
        pad.pad_u4_rows(U4RowPadArgs {
            stream,
            input: projections.qkv_weight_device.bytes,
            output: padded_weight,
            rows: dims.qkv_dim,
            padded_rows: padded,
            cols_u4: dims.embedding_dim,
        })?;
    }
    let weight = if padded == dims.qkv_dim {
        projections.qkv_weight_device.bytes
    } else {
        &*padded_weight
    };
    tma.prepare_tma_nvfp4_csf_weight_into(
        stream,
        input.bytes,
        input.scales,
        weight,
        projections.qkv_weight_device.scales,
        rows,
        dims.embedding_dim,
        padded,
        descriptors,
    )?;
    tma.gemm_tma_nvfp4_rowwise_a_scale_affine_padded_output(
        stream,
        descriptors,
        out,
        projections.qkv_bias,
        rows,
        dims.embedding_dim,
        dims.qkv_dim,
        padded,
        input.global_scales,
        projections.qkv_weight_device.global_scale,
    )
}
