mod budget;
mod pool;

use std::collections::HashSet;

use super::super::{
    analysis, candidate::Candidate, config::SweepConfig, history::Trial, rng::SweepRng,
    test_fixtures::quality_config,
};

fn changed_factors(left: &Candidate, right: &Candidate) -> usize {
    usize::from(left.batch_size != right.batch_size)
        + usize::from(left.n_layer != right.n_layer)
        + usize::from(left.n_embd != right.n_embd)
        + usize::from(left.muon_phases != right.muon_phases)
        + usize::from(left.muon_blocks != right.muon_blocks)
        + usize::from(left.lr_scale != right.lr_scale)
        + usize::from(left.adam_lr_scale != right.adam_lr_scale)
        + usize::from(left.nextlat_lr_scale != right.nextlat_lr_scale)
        + usize::from(left.warmup_steps != right.warmup_steps)
        + usize::from(left.start_ratio != right.start_ratio)
        + usize::from(left.amuse_beta1 != right.amuse_beta1)
        + usize::from(left.amuse_rho != right.amuse_rho)
}

fn sample_pool(
    seed: u64,
    config: &SweepConfig,
    trials: &[Trial],
    center: Option<&Candidate>,
) -> Vec<super::PooledCandidate> {
    let analysis = analysis::analyze(trials, config);
    let observed = trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<Vec<_>>();
    super::sample(
        &HashSet::new(),
        &mut SweepRng::new(seed),
        config,
        &analysis,
        center,
        &observed,
    )
}

fn wide_candidate(i: usize) -> Candidate {
    Candidate {
        batch_size: [4, 8, 12, 16, 20, 24, 28, 32][i % 8],
        n_layer: [4, 8][(i / 3) % 2],
        n_embd: [1024, 2048][(i / 5) % 2],
        n_head: 16,
        muon_phases: [4, 8, 16][(i / 7) % 3],
        muon_blocks: [80, 90, 120, 160, 180][(i / 11) % 5],
        lr_scale: 0.5 + (i % 11) as f64 * 0.18,
        adam_lr_scale: 0.5 + (i % 13) as f64 * 0.15,
        nextlat_lr_scale: 0.5 + (i % 17) as f64 * 0.12,
        warmup_steps: 5 + (i * 7) % 96,
        start_ratio: (i % 9) as f64 * 0.025,
        amuse_beta1: 0.2 + (i % 7) as f64 * 0.06,
        amuse_rho: 0.5 + (i % 6) as f64 * 0.08,
    }
}

fn config() -> SweepConfig {
    quality_config(8)
}
