use cuda_core::CudaStream;

use super::super::format::CheckpointWriter;
use crate::{
    AppResult,
    upload::{UploadedCanon, UploadedNvfp4},
};

pub(super) fn write(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedNvfp4,
) -> AppResult {
    writer.write_tensor(
        name,
        tensor.len,
        tensor.global_scale.to_host_vec(stream)?[0],
        &tensor.bytes.to_host_vec(stream)?,
        &tensor.scales.to_host_vec(stream)?,
    )
}

pub(super) fn write_fp32(
    writer: &mut CheckpointWriter<impl std::io::Write>,
    stream: &CudaStream,
    name: &str,
    tensor: &UploadedCanon,
) -> AppResult {
    let host = tensor.weight.to_host_vec(stream)?;
    let bytes = host
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_tensor(name, host.len(), 1.0, &bytes, &[])
}
