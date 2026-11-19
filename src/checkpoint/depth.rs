use std::{collections::HashMap, io::Write};

use gpt2_nvfp4::{GPT2_N_LAYER, KIMI_FULL_ATTENTION_PERIOD as GROWTH, depth_growth::LayerLayout};

use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;

pub(super) const NAME: &str = "model.active_depth";
const SOURCE_BLOCK_POLICY: u32 = 1;

pub(super) fn extra_tensors(layout: LayerLayout) -> u32 {
    u32::from(layout.depth != GPT2_N_LAYER || layout.inherit_block_policy)
}

pub(super) fn write(writer: &mut CheckpointWriter<impl Write>, layout: LayerLayout) -> AppResult {
    if extra_tensors(layout) == 0 {
        return Ok(());
    }
    validate_prefix(layout)?;
    let depth = (layout.depth as u32).to_le_bytes();
    if layout.inherit_block_policy {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&depth);
        bytes[4..].copy_from_slice(&SOURCE_BLOCK_POLICY.to_le_bytes());
        writer.write_tensor(NAME, 2, 1.0, &bytes, &[])
    } else {
        writer.write_tensor(NAME, 1, 1.0, &depth, &[])
    }
}

pub(super) fn read(tensor: &CheckpointTensor) -> AppResult<LayerLayout> {
    if tensor.global_scale != 1.0 || !tensor.scales.is_empty() {
        return Err("invalid checkpoint active depth descriptor".into());
    }
    let layout = match (tensor.len, tensor.bytes.as_slice()) {
        (1, [d0, d1, d2, d3]) => {
            LayerLayout::from(u32::from_le_bytes([*d0, *d1, *d2, *d3]) as usize)
        }
        (2, [d0, d1, d2, d3, p0, p1, p2, p3])
            if u32::from_le_bytes([*p0, *p1, *p2, *p3]) == SOURCE_BLOCK_POLICY =>
        {
            LayerLayout::grown(u32::from_le_bytes([*d0, *d1, *d2, *d3]) as usize)
        }
        _ => return Err("invalid checkpoint active depth descriptor".into()),
    };
    validate_prefix(layout)?;
    Ok(layout)
}

pub(super) fn take(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult<LayerLayout> {
    tensors
        .remove(NAME)
        .as_ref()
        .map_or(Ok(GPT2_N_LAYER.into()), read)
}

fn validate_prefix(layout: LayerLayout) -> AppResult {
    let depth = layout.depth;
    if (GROWTH..=GPT2_N_LAYER).contains(&depth)
        && depth.is_multiple_of(GROWTH)
        && (depth != GPT2_N_LAYER || layout.inherit_block_policy)
    {
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

    fn serialized_descriptor(layout: LayerLayout) -> AppResult<CheckpointTensor> {
        let mut data = Vec::new();
        {
            let mut writer = CheckpointWriter::new(BufWriter::new(&mut data));
            writer.write_header(extra_tensors(layout))?;
            write(&mut writer, layout)?;
            writer.finish()?;
        }
        let mut reader = CheckpointReader::new(BufReader::new(data.as_slice()));
        assert_eq!(reader.read_header()?, 1);
        let tensor = reader.read_tensor()?;
        assert_eq!(tensor.name, NAME);
        Ok(tensor)
    }

    #[test]
    fn legacy_checkpoint_keeps_positional_policy_at_partial_and_full_depth() -> AppResult {
        let fixed = GPT2_N_LAYER.into();
        assert_eq!(extra_tensors(fixed), 0);
        let mut full = Vec::new();
        write(&mut CheckpointWriter::new(BufWriter::new(&mut full)), fixed)?;
        assert!(full.is_empty());
        assert_eq!(take(&mut HashMap::new())?, fixed);

        if GPT2_N_LAYER > 4 {
            let old = LayerLayout::from(4);
            let mut tensor = serialized_descriptor(old)?;
            assert_eq!(tensor.len, 1);
            assert_eq!(tensor.bytes, 4_u32.to_le_bytes());
            assert_eq!(read(&tensor)?, old);
            tensor.bytes = 5_u32.to_le_bytes().to_vec();
            assert!(read(&tensor).is_err());
            tensor.bytes = (GPT2_N_LAYER as u32).to_le_bytes().to_vec();
            assert!(read(&tensor).is_err());
        }
        Ok(())
    }

    #[test]
    fn growth_checkpoint_preserves_source_policy_even_at_full_depth() -> AppResult {
        for depth in (GROWTH..=GPT2_N_LAYER).step_by(GROWTH) {
            let layout = LayerLayout::grown(depth);
            let mut tensor = serialized_descriptor(layout)?;
            assert_eq!(tensor.len, 2);
            assert_eq!(read(&tensor)?, layout);
            tensor.bytes[4] = 2;
            assert!(read(&tensor).is_err());
            tensor.bytes[4] = SOURCE_BLOCK_POLICY as u8;
            tensor.bytes[0] |= 1;
            assert!(read(&tensor).is_err());
            tensor.bytes[0] &= !1;
            assert_eq!(take(&mut HashMap::from([(NAME.into(), tensor)]))?, layout);
        }
        Ok(())
    }
}
