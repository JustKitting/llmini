//! Numerical MLP graph metadata; declared tests precede serialization wiring.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};

fn bounded_descriptor() -> Vec<f32> {
    // version, function, d, product, paired, caps, route geometry,
    // recomputation, joint paired parameter, stable analytic derivative,
    // represented activation-input bits and straight-through quantizer.
    vec![
        2., 1., 2048., 5504., 11008., 4., 25., 43., 32., 1., 1., 1., 16., 1.,
    ]
}
fn validate_descriptor(values: Option<&[f32]>, bounded: bool) -> AppResult {
    match values {
        None if !bounded => Ok(()),
        Some(values) if bounded && values == bounded_descriptor() => Ok(()),
        None => Err(
            "legacy ReLU checkpoint is incompatible with the compiled bounded-SwiGLU graph".into(),
        ),
        Some(_) => Err("MLP checkpoint graph/layout does not match this native executable".into()),
    }
}
pub(super) fn extra_tensors() -> u32 {
    gpt2_nvfp4::GPT2_BOUNDED_SWIGLU as u32
}
pub(super) fn write(writer: &mut CheckpointWriter<impl Write>) -> AppResult {
    if gpt2_nvfp4::GPT2_BOUNDED_SWIGLU {
        let values = bounded_descriptor();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        writer.write_tensor("mlp.config", values.len(), 1., &bytes, &[])?;
    }
    Ok(())
}
pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    let values = match tensors.remove("mlp.config") {
        None => None,
        Some(t) => {
            if t.len != 14 || t.bytes.len() != 56 || !t.scales.is_empty() || t.global_scale != 1. {
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
    validate_descriptor(values.as_deref(), gpt2_nvfp4::GPT2_BOUNDED_SWIGLU)
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
        assert_eq!(bounded_descriptor(), required);
        assert!(validate_descriptor(Some(&required), true).is_ok());
        assert!(validate_descriptor(Some(&old), true).is_err());
        assert!(validate_descriptor(None, false).is_ok());
    }
    #[test]
    fn paired_mlp_metadata_is_explicit_and_never_reinterprets_legacy_relu() {
        let descriptor = bounded_descriptor();
        assert!(validate_descriptor(Some(&descriptor), true).is_ok());
        assert!(validate_descriptor(None, false).is_ok());
        assert!(validate_descriptor(None, true).is_err());
        assert!(validate_descriptor(Some(&descriptor), false).is_err());
        for index in 0..descriptor.len() {
            let mut changed = descriptor.clone();
            changed[index] += 1.;
            assert!(
                validate_descriptor(Some(&changed), true).is_err(),
                "unchecked descriptor field{index}"
            );
        }
        let mut nonfinite = descriptor;
        nonfinite[0] = f32::NAN;
        assert!(validate_descriptor(Some(&nonfinite), true).is_err());
    }
}
