use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use cuda_core::CudaStream;
use gpt2_nvfp4::GPT2_N_LAYER;

mod tensor;
mod topology;

use super::{format::CheckpointReader, schema};
use crate::AppResult;
use crate::upload::UploadedModel;

pub fn load_uploaded_model(stream: &CudaStream, path: &Path) -> AppResult<UploadedModel> {
    let file = File::open(path)?;
    let mut reader = CheckpointReader::new(BufReader::new(file));
    let tensor_count = reader.read_header()?;
    let expected_tensor_count = schema::tensor_count(GPT2_N_LAYER);
    let legacy_tensor_count = expected_tensor_count - 1;
    if tensor_count != expected_tensor_count && tensor_count != legacy_tensor_count {
        return Err(format!(
            "checkpoint has {tensor_count} tensors; expected {expected_tensor_count} \
             (or legacy count {legacy_tensor_count})",
        )
        .into());
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let tensor = reader.read_tensor()?;
        if tensors.insert(tensor.name.clone(), tensor).is_some() {
            return Err("checkpoint contains duplicate tensor name".into());
        }
    }

    topology::load_model(stream, &mut tensors)
}
