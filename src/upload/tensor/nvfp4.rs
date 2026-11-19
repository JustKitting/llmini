use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{Nvfp4Shape, Nvfp4Tensor};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::AppResult;

pub struct UploadedNvfp4 {
    pub(crate) bytes: DeviceBuffer<u8>,
    pub(crate) scales: DeviceBuffer<u8>,
    pub(crate) global_scale: DeviceBuffer<f32>,
    pub(crate) len: usize,
    pub(crate) scale_rows: u32,
    pub(crate) scale_k: u32,
}

impl UploadedNvfp4 {
    pub(crate) fn from_host(
        stream: &CudaStream,
        bytes: &[u8],
        scales: &[u8],
        global_scale: f32,
        len: usize,
    ) -> AppResult<Self> {
        let scale_k = u32::from_le_bytes(scales[0..4].try_into().expect("CSF K header"));
        let scale_rows = u32::from_le_bytes(scales[4..8].try_into().expect("CSF row header"));
        let host_scale = [global_scale];
        let uploaded = (|| -> AppResult<Self> {
            Ok(Self {
                bytes: DeviceBuffer::from_host(stream, bytes)?,
                scales: DeviceBuffer::from_host(stream, scales)?,
                global_scale: DeviceBuffer::from_host(stream, &host_scale)?,
                len,
                scale_k,
                scale_rows,
            })
        })();
        // Complete every borrowed host copy, including the stack scale, before
        // returning. Callers may release their payload/scale slices afterward.
        // Drain earlier enqueues even if a later upload/allocation failed.
        let synchronized = stream.synchronize();
        let uploaded = uploaded?;
        synchronized?;
        Ok(uploaded)
    }

    pub(crate) fn global_scale_to_host(&self, stream: &CudaStream) -> AppResult<f32> {
        Ok(self.global_scale.to_host_vec(stream)?[0])
    }

    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }
}

pub(in crate::upload) fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> AppResult<UploadedNvfp4> {
    upload_nvfp4_k(stream, tensor, S::COLS.max(128).next_power_of_two())
}
pub(in crate::upload) fn upload_nvfp4_k<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
    k: usize,
) -> AppResult<UploadedNvfp4> {
    let scales = rust_kernels_cuda::nvfp4::csf::encode_host(
        tensor.scales.as_ref(),
        tensor.len().div_ceil(k),
        k,
    );
    let out = UploadedNvfp4::from_host(
        stream,
        tensor.bytes.as_ref(),
        &scales,
        tensor.global_scale,
        tensor.len(),
    )?;
    stream.synchronize()?;
    Ok(out)
}
