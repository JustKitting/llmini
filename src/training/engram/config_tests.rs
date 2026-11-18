//! Checkpoint graph contracts, declared before the query/key RMS fidelity repair.
//! These tests inspect metadata only and never create a CUDA context or train.
use super::{Config, LAYERS};

#[test]
fn reference_gate_rms_uses_new_checkpoint_graph_version() {
    for index in 0..LAYERS.len() {
        assert_eq!(
            Config::reference(index).descriptor()[0],
            3.,
            "new query/key RMS semantics must not reuse an old checkpoint version"
        );
    }
}

#[test]
fn reference_gate_rms_descriptor_round_trips_without_legacy_reinterpretation() {
    let mut descriptor = Config::reference(0).descriptor();
    descriptor[0] = 3.;
    let parsed = Config::parse(&descriptor, false)
        .expect("the reference query/key RMS graph must have an explicit load path");
    assert_eq!(parsed.descriptor(), descriptor);
    assert!(Config::parse(&descriptor, true).is_err());
}

#[test]
fn legacy_gate_rms_descriptors_preserve_their_exact_graph_versions() {
    for version in [1., 2.] {
        let mut descriptor = Config::reference(0).descriptor();
        descriptor[0] = version;
        if version == 1. {
            descriptor[2] = 32.;
            descriptor[3] = 512.;
        }
        let parsed = Config::parse(&descriptor, version == 1.)
            .expect("existing Engram checkpoints must retain their declared graph");
        assert_eq!(parsed.descriptor(), descriptor);
        assert_eq!(parsed.gate_rms_epsilon(), 1e-5);
    }
    let mut unknown = Config::reference(0).descriptor();
    unknown[0] = 5.;
    assert!(Config::parse(&unknown, false).is_err());
}

#[test]
fn reference_gate_rms_epsilon_matches_the_fp32_demo_default() {
    for index in 0..LAYERS.len() {
        let reference = Config::reference(index);
        assert_eq!(reference.gate_rms_epsilon(), f32::EPSILON);
        let restored = Config::parse(&reference.descriptor(), false).unwrap();
        assert_eq!(restored.gate_rms_epsilon(), f32::EPSILON);
    }
}
