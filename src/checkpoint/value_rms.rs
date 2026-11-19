//! Exact value-coordinate graph identity, checked before any model upload.
use super::format::{CheckpointTensor, CheckpointWriter};
use crate::AppResult;
use std::{collections::HashMap, io::Write};
fn descriptor() -> [f32; 14] {
    // version, enabled, head width, epsilon, post-SiLU, pre-beta,
    // raw first-value cache, exact NVFP4 recomputation, separate shared gradient.
    // Original affine/forward input32; represented reverse SiLU16/direction16;
    // unchanged scalar normalization-control32; existing STE convention.
    [
        2., 1., 64., 1e-6, 1., 1., 1., 1., 1., 32., 16., 16., 32., 1.,
    ]
}
pub(super) fn extra_tensors() -> u32 {
    gpt2_nvfp4::GPT2_VALUE_RMS as u32
}
pub(super) fn write(w: &mut CheckpointWriter<impl Write>) -> AppResult {
    if gpt2_nvfp4::GPT2_VALUE_RMS {
        let values = descriptor();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        w.write_tensor("value_rms.config", values.len(), 1., &bytes, &[])?;
    }
    Ok(())
}
pub(super) fn take_and_validate(tensors: &mut HashMap<String, CheckpointTensor>) -> AppResult {
    match tensors.remove("value_rms.config") {
        None if !gpt2_nvfp4::GPT2_VALUE_RMS => Ok(()),
        Some(t)
            if gpt2_nvfp4::GPT2_VALUE_RMS
                && t.len == 14
                && t.bytes.len() == 56
                && t.scales.is_empty()
                && t.global_scale == 1.
                && t.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .eq(descriptor()) =>
        {
            Ok(())
        }
        _ => {
            Err("checkpoint value-normalization graph does not match this native executable".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32]) -> HashMap<String, CheckpointTensor> {
        HashMap::from([(
            "value_rms.config".into(),
            CheckpointTensor {
                name: "value_rms.config".into(),
                len: values.len(),
                global_scale: 1.,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                scales: Vec::new(),
            },
        )])
    }

    #[test]
    fn value_reverse_policy_uses_half_coordinates_and_rejects_original_fp32_point() {
        let required = [
            2., 1., 64., 1e-6, 1., 1., 1., 1., 1., 32., 16., 16., 32., 1.,
        ];
        let original = [1., 1., 64., 1e-6, 1., 1., 1., 1., 1.];
        assert_eq!(descriptor().as_slice(), required.as_slice());
        assert!(take_and_validate(&mut tensor(&original)).is_err());
        assert_eq!(
            take_and_validate(&mut tensor(&required)).is_ok(),
            gpt2_nvfp4::GPT2_VALUE_RMS
        );
        assert_eq!(
            take_and_validate(&mut HashMap::new()).is_ok(),
            !gpt2_nvfp4::GPT2_VALUE_RMS
        );
        for index in 0..required.len() {
            let mut changed = required;
            changed[index] += 1.;
            assert!(
                take_and_validate(&mut tensor(&changed)).is_err(),
                "unchecked value-RMS descriptor field{index}"
            );
        }
        let mut nonfinite = required;
        nonfinite[0] = f32::NAN;
        assert!(take_and_validate(&mut tensor(&nonfinite)).is_err());
    }
}
