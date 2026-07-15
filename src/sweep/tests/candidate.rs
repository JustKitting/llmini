use crate::sweep::candidate::{
    Candidate, MIN_N_EMBD, MIN_N_HEAD, MIN_N_LAYER, valid_aurora_phases,
};

use super::fixtures::rng;

#[test]
fn exposes_profiled_l2_aurora_phase_layout() {
    assert!(valid_aurora_phases(8, 90).contains(&2));
    assert!(!valid_aurora_phases(16, 90).contains(&2));
    assert!(valid_aurora_phases(16, 90).contains(&4));
}

#[test]
fn random_candidates_respect_min_layer_count() {
    let mut rng = rng();
    for _ in 0..256 {
        let candidate = Candidate::random(&mut rng);
        assert!(candidate.n_layer >= MIN_N_LAYER);
        assert!(candidate.n_embd >= MIN_N_EMBD);
        assert!(candidate.n_head >= MIN_N_HEAD);
    }
}
