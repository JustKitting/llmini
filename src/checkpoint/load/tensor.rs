use std::collections::HashMap;

use cuda_core::CudaStream;

use super::super::format::CheckpointTensor;
use crate::{
    AppResult,
    upload::{UploadedCanon, UploadedNvfp4},
};

pub(super) fn take_uploaded(
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

pub(super) fn take_uploaded_fp32(
    stream: &CudaStream,
    tensors: &mut HashMap<String, CheckpointTensor>,
    name: &str,
) -> AppResult<UploadedCanon> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| format!("checkpoint is missing tensor {name}"))?;
    validate_len(
        name,
        "values",
        tensor.len,
        gpt2_nvfp4::GPT2_CANON_WEIGHT_COUNT,
    )?;
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

fn validate(name: &str, tensor: &CheckpointTensor) -> AppResult {
    validate_len(name, "bytes", tensor.bytes.len(), tensor.len / 2)?;
    validate_len(name, "scales", tensor.scales.len(), tensor.len / 16)
}

fn validate_len(name: &str, field: &str, actual: usize, expected: usize) -> AppResult {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{name} has {actual} {field}; expected {expected}").into())
    }
}
