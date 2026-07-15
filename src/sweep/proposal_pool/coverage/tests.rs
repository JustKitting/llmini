use super::{coverage_score, unit_features};
use crate::sweep::candidate::Candidate;

#[test]
fn coverage_score_prefers_uncovered_region() {
    let observed = [unit_features(&candidate(4, 4, 0.5, 0.5, 5, 0.0, 0.2, 0.5))];
    let near = candidate(4, 4, 0.55, 0.55, 8, 0.02, 0.22, 0.55);
    let far = candidate(16, 8, 2.5, 2.5, 100, 0.2, 0.6, 1.0);

    assert!(coverage_score(&far, &observed) > coverage_score(&near, &observed));
}

fn candidate(
    batch_size: usize,
    n_layer: usize,
    lr_scale: f64,
    adam_lr_scale: f64,
    warmup_steps: usize,
    start_ratio: f64,
    amuse_beta1: f64,
    amuse_rho: f64,
) -> Candidate {
    Candidate {
        batch_size,
        n_layer,
        n_embd: if n_layer > 4 { 2048 } else { 1024 },
        n_head: 16,
        muon_phases: if n_layer > 4 { 16 } else { 2 },
        muon_blocks: if batch_size > 4 { 180 } else { 80 },
        lr_scale,
        adam_lr_scale,
        nextlat_lr_scale: 1.0,
        warmup_steps,
        start_ratio,
        amuse_beta1,
        amuse_rho,
    }
}
