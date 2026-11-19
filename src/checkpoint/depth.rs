use std::{collections::HashMap, io::Write};

use gpt2_nvfp4::GPT2_N_LAYER;

use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;

pub(super) const NAME: &str = "model.active_depth";

pub(super) fn write(writer: &mut CheckpointWriter<impl Write>, depth: usize) -> AppResult {
    if depth == GPT2_N_LAYER {
        return Ok(());
    }
    validate_prefix(depth)?;
    writer.write_tensor(NAME, 1, 1.0, &(depth as u32).to_le_bytes(), &[])
}

pub(super) fn read(tensor: &CheckpointTensor) -> AppResult<usize> {
    if tensor.len != 1 || tensor.global_scale != 1.0 || !tensor.scales.is_empty() {
        return Err("invalid checkpoint active depth descriptor".into());
    }
    let depth = u32::from_le_bytes(tensor.bytes.as_slice().try_into()?) as usize;
    validate_prefix(depth)?;
    Ok(depth)
}

pub(super) fn take(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult<usize> {
    tensors.remove(NAME).as_ref().map_or(Ok(GPT2_N_LAYER), read)
}

fn validate_prefix(depth: usize) -> AppResult {
    if (4..GPT2_N_LAYER).contains(&depth) && depth.is_multiple_of(4) {
        Ok(())
    } else {
        Err("checkpoint active depth is not a native growth prefix".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::format::CheckpointReader;
    use std::io::{BufReader, BufWriter};

    #[test]
    fn checkpoint_prefix_round_trips_and_full_depth_keeps_the_old_layout() -> AppResult {
        let mut full = Vec::new();
        write(
            &mut CheckpointWriter::new(BufWriter::new(&mut full)),
            GPT2_N_LAYER,
        )?;
        assert!(full.is_empty());
        assert_eq!(take(&mut HashMap::new())?, GPT2_N_LAYER);

        let mut data = Vec::new();
        let mut writer = CheckpointWriter::new(BufWriter::new(&mut data));
        writer.write_header(1)?;
        write(&mut writer, 4)?;
        writer.finish()?;
        drop(writer);
        let mut reader = CheckpointReader::new(BufReader::new(data.as_slice()));
        assert_eq!(reader.read_header()?, 1);
        let mut tensor = reader.read_tensor()?;
        assert_eq!(tensor.name, NAME);
        assert_eq!(read(&tensor)?, 4);
        tensor.bytes = 5_u32.to_le_bytes().to_vec();
        assert!(read(&tensor).is_err());
        tensor.bytes = (GPT2_N_LAYER as u32).to_le_bytes().to_vec();
        assert!(read(&tensor).is_err());
        Ok(())
    }
}
