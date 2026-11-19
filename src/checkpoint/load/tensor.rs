use std::collections::HashMap;

use cuda_core::CudaStream;

use super::super::format::CheckpointTensor;
use crate::{
    upload::{UploadedCanon, UploadedNvfp4},
    AppResult,
};

pub(crate) fn take_uploaded(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
) -> AppResult<UploadedNvfp4> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("checkpoint is missing tensor {name}"))?;
    validate(name, &tensor)?;
    UploadedNvfp4::from_host(
        stream,
        &tensor.bytes,
        &tensor.scales,
        tensor.global_scale,
        tensor.len,
    )
}

pub(crate) fn take_uploaded_fp32(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
) -> AppResult<UploadedCanon> {
    take_uploaded_fp32_len(stream, tensors, name, gpt2_nvfp4::GPT2_CANON_WEIGHT_COUNT)
}

pub(crate) fn take_uploaded_fp32_len(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
    expected: usize,
) -> AppResult<UploadedCanon> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("checkpoint is missing tensor {name}"))?;
    validate_len(name, "values", tensor.len, expected)?;
    validate_len(
        name,
        "bytes",
        tensor.bytes.len(),
        tensor.len * size_of::<f32>(),
    )?;
    validate_len(name, "scales", tensor.scales.len(), 0)?;
    let values = tensor
        .bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("exact f32 byte chunk")))
        .collect::<Vec<_>>();
    UploadedCanon::from_host(stream, &values)
}

pub(super) fn validate(name: &str, tensor: &CheckpointTensor) -> AppResult {
    validate_len(name, "bytes", tensor.bytes.len(), tensor.len / 2)?;
    if tensor.scales.len() < 128 {
        return Err(format!("{name}: missing CSF header").into());
    }
    let k = u32::from_le_bytes(tensor.scales[0..4].try_into()?) as usize;
    let rows = u32::from_le_bytes(tensor.scales[4..8].try_into()?) as usize;
    if k == 0 || k % 128 != 0 || rows != tensor.len.div_ceil(k) {
        return Err(format!("{name}: invalid CSF shape").into());
    }
    validate_len(
        name,
        "scales",
        tensor.scales.len(),
        rust_kernels_cuda::nvfp4::csf::encoded_len(rows, k),
    )
}

fn validate_len(name: &str, field: &str, actual: usize, expected: usize) -> AppResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{name} has {actual} {field}; expected {expected}").into())
    }
}
