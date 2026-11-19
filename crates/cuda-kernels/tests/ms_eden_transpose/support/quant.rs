use cuda_core::{CudaStream, DeviceBuffer, DriverError};
use rust_kernels_cuda::nvfp4::Nvfp4RowwiseDeviceTensor;
use rust_kernels_cuda::nvfp4_quant::{
    QuartetBackwardMsEdenDeviceScaleQuantArgs, RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs,
};
use rust_kernels_cuda::nvfp4_tc_matmul::{
    nvfp4_tc_matmul_bytes, nvfp4_tc_matmul_chunks, nvfp4_tc_matmul_scales,
};

use super::{COLS, ROWS, SCALE_SEED, SIGN_SEED};

mod assert;

pub(in super::super) struct QuantScratch {
    pub(in super::super) bytes: DeviceBuffer<u8>,
    pub(in super::super) scales: DeviceBuffer<u8>,

    pub(in super::super) chunk_amax: DeviceBuffer<f32>,
    pub(in super::super) global_scale: DeviceBuffer<f32>,
}

impl QuantScratch {
    pub(in super::super) fn new(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_bytes(COLS as u32, ROWS as u32))?,
            scales: DeviceBuffer::zeroed(stream, nvfp4_tc_matmul_scales(COLS as u32, ROWS as u32))?,

            chunk_amax: DeviceBuffer::zeroed(
                stream,
                nvfp4_tc_matmul_chunks(COLS as u32, ROWS as u32),
            )?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(in super::super) fn new_exact(
        stream: &CudaStream,
        row_count: usize,
        row_len: usize,
    ) -> Result<Self, DriverError> {
        let element_count = row_count * row_len;

        Ok(Self {
            bytes: DeviceBuffer::zeroed(stream, element_count / 2)?,
            scales: DeviceBuffer::zeroed(
                stream,
                rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(row_count, row_len),
            )?,

            chunk_amax: DeviceBuffer::zeroed(stream, element_count / 32)?,
            global_scale: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    pub(in super::super) fn quartet_args<'a, 'out>(
        &'out mut self,
        stream: &'a CudaStream,
        x: &'a DeviceBuffer<f32>,
        row_count: usize,
        src_row_len: usize,
        dst_row_len: usize,
    ) -> QuartetBackwardMsEdenDeviceScaleQuantArgs<'a, 'out> {
        QuartetBackwardMsEdenDeviceScaleQuantArgs {
            stream,
            x,
            out_fp4: &mut self.bytes,
            out_scales: &mut self.scales,

            out_chunk_amax: &mut self.chunk_amax,
            out_global_scale: &mut self.global_scale,
            row_count: row_count as u32,
            src_row_len: src_row_len as u32,
            dst_row_len: dst_row_len as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        }
    }

    pub(in super::super) fn rowwise_transpose_args<'a, 'out>(
        &'out mut self,
        stream: &'a CudaStream,
        input: Nvfp4RowwiseDeviceTensor<'a>,
        source_rows: usize,
        source_cols: usize,
        dst_row_len: usize,
    ) -> RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs<'a, 'out> {
        RowwiseNvfp4TransposeMsEdenDeviceScaleQuantArgs {
            stream,
            input,
            out_fp4: &mut self.bytes,
            out_scales: &mut self.scales,

            out_chunk_amax: &mut self.chunk_amax,
            out_global_scale: &mut self.global_scale,
            source_rows: source_rows as u32,
            source_cols: source_cols as u32,
            dst_row_len: dst_row_len as u32,
            sign_seed: SIGN_SEED,
            scale_seed: SCALE_SEED,
        }
    }
}
