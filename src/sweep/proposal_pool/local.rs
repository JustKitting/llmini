use std::collections::HashSet;

use super::super::{
    candidate::{Candidate, valid_muon_phases},
    candidate_space as space,
    rng::SweepRng,
};

pub fn candidates(
    used: &HashSet<String>,
    rng: &mut SweepRng,
    center: Option<&Candidate>,
    count: usize,
) -> Vec<Candidate> {
    let Some(center) = center else {
        return Vec::new();
    };
    let mut seen = used.clone();
    let mut out = Vec::new();
    let mut attempts = 0;
    while out.len() < count && attempts < count.saturating_mul(64).max(64) {
        let mut candidate = center.clone();
        if attempts % 4 == 3 {
            candidate.batch_size = nearby_batch(center.batch_size, rng);
        }
        if attempts % 8 == 7 {
            let blocks = space::valid_muon_blocks(candidate.n_layer);
            candidate.muon_blocks = rng.choose(&blocks);
            candidate.muon_phases = nearby_phase(&candidate, rng);
        }
        candidate.lr_scale = jitter_log(center.lr_scale, rng, 0.35);
        candidate.adam_lr_scale = jitter_log(center.adam_lr_scale, rng, 0.35);
        candidate.nextlat_lr_scale = jitter_log(center.nextlat_lr_scale, rng, 0.35);
        candidate.warmup_steps = jitter_usize(center.warmup_steps, rng, 24);
        candidate.start_ratio = jitter_f64(center.start_ratio, rng, 0.06, space::START_RATIO_RANGE);
        candidate.amuse_beta1 = jitter_f64(center.amuse_beta1, rng, 0.08, space::AMUSE_BETA1_RANGE);
        candidate.amuse_rho = jitter_f64(center.amuse_rho, rng, 0.12, space::AMUSE_RHO_RANGE);

        if seen.insert(candidate.key()) {
            out.push(candidate);
        }
        attempts += 1;
    }
    out
}

fn nearby_batch(center: usize, rng: &mut SweepRng) -> usize {
    nearby_value(&space::BATCH_SIZE, center, rng).expect("batch size candidates are nonempty")
}

fn nearby_phase(candidate: &Candidate, rng: &mut SweepRng) -> usize {
    let phases = valid_muon_phases(candidate.n_layer * 4, candidate.muon_blocks);
    nearby_value(&phases, candidate.muon_phases, rng).unwrap_or(candidate.muon_phases)
}

fn nearby_value(values: &[usize], center: usize, rng: &mut SweepRng) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let Some(index) = values.iter().position(|value| *value == center) else {
        return Some(rng.choose(values));
    };
    let lo = index.saturating_sub(1);
    let hi = (index + 1).min(values.len() - 1);
    Some(values[lo + rng.usize(hi - lo + 1)])
}

fn jitter_log(value: f64, rng: &mut SweepRng, radius: f64) -> f64 {
    let offset = (rng.f64() - 0.5) * 2.0 * radius;
    (value * offset.exp()).clamp(space::LR_SCALE_RANGE.0, space::LR_SCALE_RANGE.1)
}

fn jitter_usize(value: usize, rng: &mut SweepRng, radius: usize) -> usize {
    let span = radius * 2 + 1;
    let offset = rng.usize(span) as isize - radius as isize;
    (value as isize + offset).clamp(
        space::WARMUP_STEPS_RANGE.0 as isize,
        space::WARMUP_STEPS_RANGE.1 as isize,
    ) as usize
}

fn jitter_f64(value: f64, rng: &mut SweepRng, radius: f64, range: (f64, f64)) -> f64 {
    let offset = (rng.f64() - 0.5) * 2.0 * radius;
    (value + offset).clamp(range.0, range.1)
}
