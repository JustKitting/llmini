use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_QKV, GPT2_TOKEN_ROWS, GPT2_VOCAB_SIZE};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    scale_layout::{sm120_scale_packed_len, sm120_scale_padded_mn_extent},
    tma::TmaNvfp4DeviceScaleDescriptors,
};

pub(super) struct ForwardTmaBuffers {
    pub(super) input_scales: DeviceBuffer<u8>,
    pub(super) wide_input_scales: DeviceBuffer<u8>,
    pub(super) weight_scales: DeviceBuffer<u8>,
    pub(super) weight_bytes_padded: DeviceBuffer<u8>,
    pub(super) descriptors: TmaNvfp4DeviceScaleDescriptors,
}

impl ForwardTmaBuffers {
    pub(super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        let rows = sm120_scale_padded_mn_extent(GPT2_TOKEN_ROWS);
        Ok(Self {
            input_scales: DeviceBuffer::zeroed(stream, sm120_scale_packed_len(rows, GPT2_N_EMBD))?,
            wide_input_scales: DeviceBuffer::zeroed(
                stream,
                sm120_scale_packed_len(rows, GPT2_MLP),
            )?,
            weight_scales: DeviceBuffer::zeroed(
                stream,
                sm120_scale_packed_len(sm120_scale_padded_mn_extent(GPT2_VOCAB_SIZE), GPT2_N_EMBD),
            )?,
            weight_bytes_padded: DeviceBuffer::zeroed(
                stream,
                sm120_scale_padded_mn_extent(GPT2_QKV) * GPT2_N_EMBD / 2,
            )?,
            descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
        })
    }
}
