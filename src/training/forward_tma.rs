use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_N_EMBD, GPT2_QKV};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    scale_layout::sm120_scale_padded_mn_extent, tma::TmaNvfp4DeviceScaleDescriptors,
};

pub(super) struct ForwardTmaBuffers {
    pub(super) weight_bytes_padded: DeviceBuffer<u8>,
    pub(super) descriptors: TmaNvfp4DeviceScaleDescriptors,
}

impl ForwardTmaBuffers {
    pub(super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            weight_bytes_padded: DeviceBuffer::zeroed(
                stream,
                sm120_scale_padded_mn_extent(GPT2_QKV) * GPT2_N_EMBD / 2,
            )?,
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
        })
    }
}
