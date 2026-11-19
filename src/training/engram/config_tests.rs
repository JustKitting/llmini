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

#[test]
fn second_engram_placement_preserves_the_two_modules_and_their_hashes() -> crate::AppResult {
    use super::MultistreamOptions;

    for four_grams in [false, true] {
        let control = MultistreamOptions {
            requested_values: Some(500_000_000),
            four_grams,
            ..MultistreamOptions::default()
        };
        let moved = MultistreamOptions { second_layer: Some(8), ..control }.configs()?;
        for (index, (before, after)) in control.configs()?.into_iter().zip(moved).enumerate() {
            assert_eq!(after.layer, [1, 8][index]);
            assert_eq!(before.primes, after.primes);
            assert_eq!(before.multipliers, after.multipliers);
            assert_eq!(before.initialization_layer(), after.initialization_layer());
            assert_eq!(before.table_len(), after.table_len());
            assert_eq!(before.trainable_parameters(), after.trainable_parameters());
            let read = Config::parse(&after.descriptor(), false)?;
            assert_eq!(read.layer, after.layer);
            assert_eq!(read.descriptor(), after.descriptor());
        }
        for invalid in [0, 1, gpt2_nvfp4::GPT2_N_LAYER] {
            assert!(MultistreamOptions { second_layer: Some(invalid), ..control }.configs().is_err());
        }
    }
    Ok(())
}
