use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;

use super::scratch::RowwiseNvfp4Scratch;
use crate::GPT2_TOKEN_ROWS;
use crate::types::tape::RowwiseNvfp4Tape;

pub struct RowwiseNvfp4Buffers {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scales: DeviceBuffer<f32>,
}

impl RowwiseNvfp4Buffers {
    pub fn new(
        stream: &CudaStream,
        elements: usize,
        row_count: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, elements / 2)?,
            scales: DeviceBuffer::zeroed(
                stream,
                rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(row_count, elements / row_count),
            )?,
            global_scales: DeviceBuffer::zeroed(stream, row_count)?,
        })
    }

    pub fn gpt2_rows(stream: &CudaStream, elements: usize) -> Result<Self, DriverError> {
        Self::new(stream, elements, GPT2_TOKEN_ROWS)
    }

    pub fn scratch(&mut self) -> RowwiseNvfp4Scratch<'_> {
        RowwiseNvfp4Scratch {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }

    pub fn tape(&mut self) -> RowwiseNvfp4Tape<'_> {
        RowwiseNvfp4Tape {
            bytes: &mut self.bytes,
            scales: &mut self.scales,
            global_scales: &mut self.global_scales,
        }
    }

    pub fn rowwise(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.bytes, &self.scales, &self.global_scales)
    }

    pub fn saved(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        self.rowwise()
    }
}
