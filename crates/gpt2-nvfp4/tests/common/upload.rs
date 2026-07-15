#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer};
use gpt2_nvfp4::{
    AttentionProjectionTensors, LayerNormTensors, LayerNormWeights, MlpDownTensors,
    MlpProjectionTensors, MlpUpTensors, Nvfp4Shape, Nvfp4Tensor,
};
use rust_kernels_cuda::mma::Nvfp4FourSixMmaWeightTensor;
use rust_kernels_cuda::nvfp4::Nvfp4DeviceTensor;

use crate::common::nvfp4::E4M3_ONE;

pub mod block;

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct UploadedNvfp4 {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
}

impl UploadedNvfp4 {
    pub fn device(&self) -> Nvfp4DeviceTensor<'_> {
        Nvfp4DeviceTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }

    pub fn mma(&self) -> Nvfp4FourSixMmaWeightTensor<'_> {
        Nvfp4FourSixMmaWeightTensor::new(&self.bytes, &self.scales, &self.global_scale)
    }
}

pub fn attention_projection_tensors<'a>(
    qkv_weight: &'a UploadedNvfp4,
    qkv_bias: &'a UploadedNvfp4,
    c_proj_weight: &'a UploadedNvfp4,
    c_proj_bias: &'a UploadedNvfp4,
) -> AttentionProjectionTensors<'a> {
    AttentionProjectionTensors {
        qkv_weight: qkv_weight.mma(),
        qkv_weight_device: qkv_weight.device(),
        qkv_bias: qkv_bias.device(),
        c_proj_weight: c_proj_weight.mma(),
        c_proj_weight_device: c_proj_weight.device(),
        c_proj_bias: c_proj_bias.device(),
    }
}

pub fn mlp_projection_tensors<'a>(
    up_weight: &'a UploadedNvfp4,
    up_bias: &'a UploadedNvfp4,
    down_weight: &'a UploadedNvfp4,
    down_bias: &'a UploadedNvfp4,
) -> MlpProjectionTensors<'a> {
    MlpProjectionTensors {
        up: MlpUpTensors {
            weight: up_weight.mma(),
            weight_device: up_weight.device(),
            bias: up_bias.device(),
        },
        down: MlpDownTensors {
            weight: down_weight.mma(),
            weight_device: down_weight.device(),
            bias: down_bias.device(),
        },
    }
}

pub fn layer_norm_tensors<'a>(
    weight: &'a UploadedNvfp4,
    bias: &'a UploadedNvfp4,
) -> LayerNormTensors<'a> {
    LayerNormTensors {
        weight: weight.device(),
        bias: bias.device(),
    }
}

pub struct UploadedPair {
    pub weight: UploadedNvfp4,
    pub bias: UploadedNvfp4,
}

pub type UploadedLayerNorm = UploadedPair;
pub type UploadedLinear = UploadedPair;

impl UploadedPair {
    pub fn tensors(&self) -> LayerNormTensors<'_> {
        layer_norm_tensors(&self.weight, &self.bias)
    }
}

pub fn upload_layer_norm(
    stream: &CudaStream,
    layer_norm: &LayerNormWeights,
) -> TestResult<UploadedLayerNorm> {
    Ok(UploadedPair {
        weight: upload_nvfp4(stream, &layer_norm.weight)?,
        bias: upload_nvfp4(stream, &layer_norm.bias)?,
    })
}

pub fn upload_nvfp4<S: Nvfp4Shape>(
    stream: &CudaStream,
    tensor: &Nvfp4Tensor<S>,
) -> TestResult<UploadedNvfp4> {
    upload_nvfp4_parts(
        stream,
        tensor.bytes.as_ref(),
        tensor.scales.as_ref(),
        tensor.global_scale,
    )
}

pub fn upload_nvfp4_bytes<S: Nvfp4Shape>(
    stream: &CudaStream,
    bytes: Vec<u8>,
) -> TestResult<UploadedNvfp4> {
    assert_eq!(bytes.len(), S::BYTE_LEN);
    upload_nvfp4_parts(stream, &bytes, &vec![E4M3_ONE; S::SCALE_LEN], 1.0)
}

fn upload_nvfp4_parts(
    stream: &CudaStream,
    bytes: &[u8],
    scales: &[u8],
    global_scale: f32,
) -> TestResult<UploadedNvfp4> {
    Ok(UploadedNvfp4 {
        bytes: DeviceBuffer::from_host(stream, bytes)?,
        scales: DeviceBuffer::from_host(stream, scales)?,
        global_scale: DeviceBuffer::from_host(stream, &[global_scale])?,
    })
}

pub fn upload_zero_nvfp4<S: Nvfp4Shape>(stream: &CudaStream) -> TestResult<UploadedNvfp4> {
    upload_nvfp4_bytes::<S>(stream, vec![0; S::BYTE_LEN])
}
