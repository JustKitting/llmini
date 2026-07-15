use crate::sweep::{
    candidate::{MIN_N_EMBD, MIN_N_HEAD, MIN_N_LAYER},
    chain,
};

use super::super::fixtures::{candidate, config, measured_candidate, trial};
use super::{propose, propose_with_seen};

#[test]
fn starts_fresh_sweep_from_best_measured_baseline() {
    let measured = measured_candidate();
    let config = config(3, 32);
    let baseline = propose(&[], &config, Some(&measured)).candidate;

    assert_eq!(baseline.batch_size, 8);
    assert_eq!(baseline.n_layer, MIN_N_LAYER);
    assert_eq!(baseline.n_embd, MIN_N_EMBD);
    assert_eq!(baseline.n_head, MIN_N_HEAD);
    assert_eq!(baseline.aurora_phases, 16);
    assert_eq!(baseline.aurora_blocks, 80);
    assert_eq!(baseline.lr_scale, 1.014_040);
    assert_eq!(baseline.adam_lr_scale, 1.980_467);
    assert_eq!(baseline.nextlat_lr_scale, 1.0);
    assert_eq!(baseline.warmup_steps, 5);
    assert_eq!(baseline.start_ratio, 0.05);
    assert_eq!(baseline.amuse_beta1, 0.2);
    assert_eq!(baseline.amuse_rho, 0.5);
}

#[test]
fn measured_baseline_is_not_rerun_as_next_candidate() {
    let seed_history = [trial("success", Some(1.0), candidate(8, 2, 1.0))];
    let baseline_candidate = candidate(8, 4, 2.0);
    let baseline_trial = trial("success", Some(4.2), baseline_candidate.clone());
    let all_trials = chain::all_trials_with_baseline(Some(&baseline_trial), &seed_history, &[]);
    let seen = chain::seen_keys(&all_trials);
    let config = config(1, 16);
    let proposal = propose_with_seen(&all_trials, &seen, &config, Some(&baseline_candidate));

    assert_ne!(proposal.candidate.key(), baseline_candidate.key());
    assert!(proposal.candidate.n_layer >= MIN_N_LAYER);
    assert!(proposal.candidate.n_embd >= MIN_N_EMBD);
    assert!(proposal.candidate.n_head >= MIN_N_HEAD);
}
