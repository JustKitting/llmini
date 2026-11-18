//! Descriptor semantics declared before implementation; no CUDA model involved.
use super::{format::CheckpointTensor, stationary_rope::take_and_validate};
use std::collections::HashMap;
fn tensor(values: &[f32]) -> CheckpointTensor {
    CheckpointTensor {
        name: "position.config".into(),
        len: values.len(),
        global_scale: 1.,
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        scales: vec![],
    }
}
#[test]
fn position_checkpoint_requires_exact_graph_or_legacy_default() {
    let candidate = gpt2_nvfp4::GPT2_STATIONARY_ROPE;
    let mut missing = HashMap::new();
    assert_eq!(take_and_validate(&mut missing).is_ok(), !candidate);
    let values = [1., 64., 32., 1024., 1., -1., 32., 64., 1.];
    let mut valid = HashMap::from([("position.config".into(), tensor(&values))]);
    assert_eq!(take_and_validate(&mut valid).is_ok(), candidate);
    let mut wrong = values;
    wrong[5] = 1.;
    let mut wrong = HashMap::from([("position.config".into(), tensor(&wrong))]);
    assert!(take_and_validate(&mut wrong).is_err());
}
