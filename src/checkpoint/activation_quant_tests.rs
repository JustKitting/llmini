//! Declared before policy implementation: saved policy must not silently change.
use super::{activation_quant, format::CheckpointTensor};
use std::collections::HashMap;

#[test]
fn activation_policy_descriptor_requires_exact_mae_graph_or_legacy_six() {
    let enabled = rust_kernels_cuda::nvfp4_quant::ACTIVATION_MAE_FOUR_SIX;
    assert_eq!(activation_quant::extra_tensors(), enabled as u32);
    assert_eq!(
        activation_quant::take_and_validate(&mut HashMap::new()).is_ok(),
        !enabled
    );
    // Version1, MAE selector1, group16, payload4, local E4M3, outer max256,
    // strict-less tie-to-six, activation-only; unchanged weight/gradient policy.
    let values = [1f32, 1., 16., 4., 8., 256., 1., 1.];
    for changed in 0..=values.len() {
        let mut actual = values;
        if changed < values.len() {
            actual[changed] += 1.;
        }
        let mut tensors = HashMap::from([(
            "activation_quant.config".into(),
            CheckpointTensor {
                name: "activation_quant.config".into(),
                len: actual.len(),
                global_scale: 1.,
                bytes: actual.iter().flat_map(|v| v.to_le_bytes()).collect(),
                scales: vec![],
            },
        )]);
        assert_eq!(
            activation_quant::take_and_validate(&mut tensors).is_ok(),
            enabled && changed == values.len(),
            "changed={changed}"
        );
    }
}
