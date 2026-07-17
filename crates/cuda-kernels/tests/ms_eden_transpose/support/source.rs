use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::{Nvfp4DeviceTensor, Nvfp4RowwiseDeviceTensor};
use rust_kernels_cuda::nvfp4_quant::{Nvfp4QuantModule, Nvfp4QuantRowwiseArgs, RowAmaxArgs};

use super::{COLS, ROWS};

pub(in super::super) struct SourceScratch {
    pub(in super::super) bytes: DeviceBuffer<u8>,
    pub(in super::super) scales: DeviceBuffer<u8>,
    pub(in super::super) global_scale: DeviceBuffer<f32>,
}

impl SourceScratch {
    pub(in super::super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Self::new_for_shape(stream, ROWS, COLS)
    }

    pub(in super::super) fn new_for_shape(
        stream: &CudaStream,
        rows: usize,
        cols: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, rows * cols / 2)?,
            scales: DeviceBuffer::zeroed(stream, rows * cols / 16)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(in super::super) fn tensor(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }
}

pub(in super::super) struct RowwiseSourceScratch {
    pub(in super::super) bytes: DeviceBuffer<u8>,
    pub(in super::super) scales: DeviceBuffer<u8>,
    pub(in super::super) global_scales: DeviceBuffer<f32>,
}

impl RowwiseSourceScratch {
    pub(in super::super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Self::new_for_shape(stream, ROWS, COLS)
    }

    pub(in super::super) fn new_for_shape(
        stream: &CudaStream,
        rows: usize,
        cols: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, rows * cols / 2)?,
            scales: DeviceBuffer::zeroed(stream, rows * cols / 16)?,
            global_scales: DeviceBuffer::zeroed(stream, rows)?,
        })
    }

    pub(in super::super) fn quantize(
        &mut self,
        stream: &CudaStream,
        quant: &Nvfp4QuantModule,
        x: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), DriverError> {
        let mut row_amax = DeviceBuffer::<f32>::zeroed(stream, rows)?;
        quant.row_amax_f32(RowAmaxArgs {
            stream,
            x,
            out: &mut row_amax,
            row_count: rows as u32,
            row_len: cols as u32,
        })?;
        quant.fp32_to_nvfp4_four_six_rowwise(Nvfp4QuantRowwiseArgs {
            stream,
            x,
            amax: &row_amax,
            out_fp4: &mut self.bytes,
            out_scales: &mut self.scales,
            out_global_scale: &mut self.global_scales,
            group_count: (rows * cols / 16) as u32,
            row_len: cols as u32,
        })
    }

    pub(in super::super) fn tensor(&self) -> Nvfp4RowwiseDeviceTensor<'_> {
        Nvfp4RowwiseDeviceTensor::new(&self.bytes, &self.scales, &self.global_scales)
    }
}
