//! Numerical MLP graph metadata; declared tests precede serialization wiring.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};

const BOUNDED: &[f32] = &[
    2., 1., 2048., 5504., 11008., 4., 25., 43., 32., 1., 1., 1., 16., 1.,
];
// Version, function ID, feature width, token-local centering, tape bits,
// mask direct down adjoints, and retain the dense coupled up adjoint.
const ADAPTIVE: &[f32] = &[1., 2., gpt2_nvfp4::GPT2_MLP as f32, 1., 16., 1., 1.];

fn compiled_descriptor() -> Option<&'static [f32]> {
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        Some(BOUNDED)
    } else if gpt2_nvfp4::GPT2_ADAPTIVE_RELU2 {
        Some(ADAPTIVE)
    } else {
        None
    }
}

fn validate_descriptor(values: Option<&[f32]>, expected: Option<&[f32]>) -> AppResult {
    if values == expected {
        Ok(())
    } else {
        Err("MLP checkpoint graph/layout does not match this native executable".into())
    }
}

pub(super) fn extra_tensors() -> u32 {
    u32::from(compiled_descriptor().is_some())
}
pub(super) fn write(writer: &mut CheckpointWriter<impl Write>) -> AppResult {
    if let Some(values) = compiled_descriptor() {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_tensor("mlp.config", values.len(), 1., &bytes, &[])?;
    }
    Ok(())
}
pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    let expected = compiled_descriptor();
    let values = match tensors.remove("mlp.config") {
        None => None,
        Some(t) => {
            if expected.is_none_or(|values| t.len != values.len())
                || t.bytes.len() != t.len * size_of::<f32>()
                || !t.scales.is_empty()
                || t.global_scale != 1.
            {
                return Err("invalid native MLP graph descriptor".into());
            }
            Some(
                t.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            )
        }
    };
    validate_descriptor(values.as_deref(), expected)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_policy_requires_half_inputs_and_rejects_the_untrained_fp32_prototype() {
        let required = vec![
            2., 1., 2048., 5504., 11008., 4., 25., 43., 32., 1., 1., 1., 16., 1.,
        ];
        let old = vec![1., 1., 2048., 5504., 11008., 4., 25., 43., 32., 1., 1., 1.];
        assert_eq!(BOUNDED, required);
        assert!(validate_descriptor(Some(&required), Some(BOUNDED)).is_ok());
        assert!(validate_descriptor(Some(&old), Some(BOUNDED)).is_err());
        assert!(validate_descriptor(None, None).is_ok());
    }
    #[test]
    fn paired_mlp_metadata_is_explicit_and_never_reinterprets_legacy_relu() {
        let descriptor = BOUNDED;
        assert!(validate_descriptor(Some(descriptor), Some(BOUNDED)).is_ok());
        assert!(validate_descriptor(None, None).is_ok());
        assert!(validate_descriptor(None, Some(BOUNDED)).is_err());
        assert!(validate_descriptor(Some(descriptor), None).is_err());
        for index in 0..descriptor.len() {
            let mut changed = descriptor.to_vec();
            changed[index] += 1.;
            assert!(
                validate_descriptor(Some(&changed), Some(BOUNDED)).is_err(),
                "unchecked descriptor field{index}"
            );
        }
        let mut nonfinite = descriptor.to_vec();
        nonfinite[0] = f32::NAN;
        assert!(validate_descriptor(Some(&nonfinite), Some(BOUNDED)).is_err());
    }
    #[test]
    fn adaptive_relu2_has_distinct_graph_metadata() {
        assert!(validate_descriptor(Some(ADAPTIVE), Some(ADAPTIVE)).is_ok());
        assert!(validate_descriptor(None, Some(ADAPTIVE)).is_err());
        assert!(validate_descriptor(Some(ADAPTIVE), None).is_err());
        assert!(validate_descriptor(Some(BOUNDED), Some(ADAPTIVE)).is_err());
        assert!(validate_descriptor(Some(ADAPTIVE), Some(BOUNDED)).is_err());
    }

    #[test]
    fn compiled_mlp_metadata_round_trips_native_writer() -> AppResult {
        use crate::checkpoint::CheckpointReader;
        use std::io::{BufReader, BufWriter};

        let mut bytes = Vec::new();
        let mut writer = CheckpointWriter::new(BufWriter::new(&mut bytes));
        writer.write_header(extra_tensors())?;
        write(&mut writer)?;
        writer.finish()?;
        drop(writer);

        let mut reader = CheckpointReader::new(BufReader::new(bytes.as_slice()));
        let mut records = HashMap::new();
        let count = reader.read_header()?;
        assert_eq!(count, extra_tensors());
        for _ in 0..count {
            let tensor = reader.read_tensor()?;
            records.insert(tensor.name.clone(), tensor);
        }
        take_and_validate(&mut records)?;
        assert!(records.is_empty());
        Ok(())
    }
}
