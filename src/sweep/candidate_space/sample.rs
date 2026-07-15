use super::{
    AMUSE_BETA1_RANGE, AMUSE_RHO_RANGE, BATCH_SIZE, LR_SCALE_RANGE, MUON_BLOCKS, N_EMBD, N_LAYER,
    START_RATIO_RANGE, WARMUP_STEPS_RANGE,
};
use crate::sweep::{candidate::Candidate, rng::SweepRng};

const MUON_PHASES: [usize; 4] = [2, 4, 8, 16];

pub(in crate::sweep) fn random(rng: &mut SweepRng) -> Candidate {
    let (n_embd, n_head) = rng.choose(&N_EMBD);
    let n_layer = rng.choose(&N_LAYER);
    let blocks = valid_muon_blocks(n_layer);
    let muon_blocks = rng.choose(&blocks);
    let phases = valid_muon_phases(n_layer * 4, muon_blocks);
    Candidate {
        batch_size: rng.choose(&BATCH_SIZE),
        n_layer,
        n_embd,
        n_head,
        muon_phases: rng.choose(&phases),
        muon_blocks,
        lr_scale: rng.log_uniform(LR_SCALE_RANGE.0, LR_SCALE_RANGE.1),
        adam_lr_scale: rng.log_uniform(LR_SCALE_RANGE.0, LR_SCALE_RANGE.1),
        nextlat_lr_scale: rng.log_uniform(LR_SCALE_RANGE.0, LR_SCALE_RANGE.1),
        warmup_steps: random_usize_range(rng, WARMUP_STEPS_RANGE),
        start_ratio: random_f64_range(rng, START_RATIO_RANGE),
        amuse_beta1: random_f64_range(rng, AMUSE_BETA1_RANGE),
        amuse_rho: random_f64_range(rng, AMUSE_RHO_RANGE),
    }
}

pub(in crate::sweep) fn valid_muon_blocks(n_layer: usize) -> Vec<usize> {
    MUON_BLOCKS
        .into_iter()
        .filter(|blocks| !valid_muon_phases(n_layer * 4, *blocks).is_empty())
        .collect()
}

pub(in crate::sweep) fn valid_muon_phases(slots: usize, blocks: usize) -> Vec<usize> {
    MUON_PHASES
        .into_iter()
        .filter(|phase| {
            slots.is_multiple_of(*phase) && cooperative_blocks(slots, *phase, blocks) <= 360
        })
        .collect()
}

fn cooperative_blocks(slots: usize, phases: usize, blocks: usize) -> usize {
    blocks * (slots / phases)
}

fn random_f64_range(rng: &mut SweepRng, range: (f64, f64)) -> f64 {
    range.0 + (range.1 - range.0) * rng.f64()
}

fn random_usize_range(rng: &mut SweepRng, range: (usize, usize)) -> usize {
    range.0 + rng.usize(range.1 - range.0 + 1)
}
