use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use gpt2_nvfp4::{
    GPT2_TOKEN_ROWS, HiddenState, NEXTLAT_HIDDEN, NEXTLAT_INPUT, NextLatHiddenActivation,
    NextLatInputActivation, RowwiseNvfp4Buffers, RowwiseNvfp4Scratch,
};
use rust_kernels_cuda::nvfp4_tma_matmul::{
    scale_layout::{sm120_scale_packed_len, sm120_scale_padded_mn_extent},
    tma::TmaNvfp4DeviceScaleDescriptors,
};

use super::super::device_buffer::zero;

pub struct NextLatBuffers {
    // Reused for the final predicted state after concat consumes the embedding.
    pub(super) next_token_embeddings: DeviceBuffer<f32>,
    pub(super) concat: DeviceBuffer<f32>,
    // Reused for the first pre-GELU tape after input quantization consumes it.
    pub(super) normalized: DeviceBuffer<f32>,
    pub(super) normalized_amax: DeviceBuffer<f32>,
    pub(super) mean: DeviceBuffer<f32>,
    pub(super) inv_std: DeviceBuffer<f32>,
    pub(super) input_quant: RowwiseNvfp4Buffers,
    // Reused for the second pre-GELU tape after act1 quantization consumes it.
    pub(super) act1: DeviceBuffer<f32>,
    pub(super) act1_quant: RowwiseNvfp4Buffers,
    // Reused for the output delta after act2 quantization consumes it.
    pub(super) act2: DeviceBuffer<f32>,
    pub(super) act2_quant: RowwiseNvfp4Buffers,
    pub(super) losses: DeviceBuffer<f32>,
    pub(super) d_predicted: DeviceBuffer<f32>,
    pub(super) tma_input_scale_packed: DeviceBuffer<u8>,
    pub(super) tma_weight_scale_packed: DeviceBuffer<u8>,
    pub(super) tma_descriptors: TmaNvfp4DeviceScaleDescriptors,
}

pub(super) struct RowwiseQuantizeBuffers<'a> {
    pub input: &'a DeviceBuffer<f32>,
    pub amax: &'a mut DeviceBuffer<f32>,
    pub out: RowwiseNvfp4Scratch<'a>,
}

impl NextLatBuffers {
    pub fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            next_token_embeddings: zero(stream, HiddenState::LEN)?,
            concat: zero(stream, NextLatInputActivation::LEN)?,
            normalized: zero(stream, NextLatInputActivation::LEN)?,
            normalized_amax: zero(stream, GPT2_TOKEN_ROWS)?,
            mean: zero(stream, GPT2_TOKEN_ROWS)?,
            inv_std: zero(stream, GPT2_TOKEN_ROWS)?,
            input_quant: RowwiseNvfp4Buffers::gpt2_rows(stream, NextLatInputActivation::LEN)?,
            act1: zero(stream, NextLatHiddenActivation::LEN)?,
            act1_quant: RowwiseNvfp4Buffers::gpt2_rows(stream, NextLatHiddenActivation::LEN)?,
            act2: zero(stream, NextLatHiddenActivation::LEN)?,
            act2_quant: RowwiseNvfp4Buffers::gpt2_rows(stream, NextLatHiddenActivation::LEN)?,
            losses: zero(stream, GPT2_TOKEN_ROWS)?,
            d_predicted: zero(stream, HiddenState::LEN)?,
            tma_input_scale_packed: DeviceBuffer::zeroed(
                stream,
                sm120_scale_packed_len(
                    sm120_scale_padded_mn_extent(GPT2_TOKEN_ROWS),
                    NEXTLAT_INPUT,
                ),
            )?,
            tma_weight_scale_packed: DeviceBuffer::zeroed(
                stream,
                sm120_scale_packed_len(
                    sm120_scale_padded_mn_extent(NEXTLAT_HIDDEN),
                    NEXTLAT_HIDDEN,
                ),
            )?,
            tma_descriptors: TmaNvfp4DeviceScaleDescriptors::new(stream)?,
        })
    }

    pub(crate) fn losses(&self) -> &DeviceBuffer<f32> {
        &self.losses
    }

    pub(super) fn input_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers::new(
            &self.normalized,
            &mut self.normalized_amax,
            &mut self.input_quant,
        )
    }

    pub(super) fn act1_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers::new(&self.act1, &mut self.normalized_amax, &mut self.act1_quant)
    }

    pub(super) fn act2_quantize(&mut self) -> RowwiseQuantizeBuffers<'_> {
        RowwiseQuantizeBuffers::new(&self.act2, &mut self.normalized_amax, &mut self.act2_quant)
    }
}

impl<'a> RowwiseQuantizeBuffers<'a> {
    fn new(
        input: &'a DeviceBuffer<f32>,
        amax: &'a mut DeviceBuffer<f32>,
        out: &'a mut RowwiseNvfp4Buffers,
    ) -> Self {
        Self {
            input,
            amax,
            out: out.scratch(),
        }
    }
}
