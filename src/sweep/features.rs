use super::{candidate::Candidate, candidate_space};

pub const FEATURE_COUNT: usize = 12;

pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "batch_size",
    "n_layer",
    "n_embd",
    "muon_phases",
    "muon_blocks",
    "ln_lr_scale",
    "ln_adam_lr_scale",
    "ln_nextlat_lr_scale",
    "ln_warmup_steps",
    "start_ratio",
    "amuse_beta1",
    "amuse_rho",
];

pub fn regression_features(candidate: &Candidate) -> [f64; FEATURE_COUNT] {
    [
        candidate.batch_size as f64,
        candidate.n_layer as f64,
        candidate.n_embd as f64 / 1024.0,
        candidate.muon_phases as f64,
        candidate.muon_blocks as f64 / 80.0,
        candidate.lr_scale.ln(),
        candidate.adam_lr_scale.ln(),
        candidate.nextlat_lr_scale.ln(),
        (candidate.warmup_steps as f64).ln(),
        candidate.start_ratio,
        candidate.amuse_beta1,
        candidate.amuse_rho,
    ]
}

pub fn unit_features(candidate: &Candidate) -> [f64; FEATURE_COUNT] {
    [
        range(candidate.batch_size as f64, 1.0, 32.0),
        range(candidate.n_layer as f64, 4.0, 8.0),
        range(candidate.n_embd as f64, 1024.0, 2048.0),
        range(candidate.muon_phases as f64, 2.0, 16.0),
        range(candidate.muon_blocks as f64, 80.0, 180.0),
        log_range(candidate.lr_scale, candidate_space::LR_SCALE_RANGE),
        log_range(candidate.adam_lr_scale, candidate_space::LR_SCALE_RANGE),
        log_range(candidate.nextlat_lr_scale, candidate_space::LR_SCALE_RANGE),
        range_usize(candidate.warmup_steps, candidate_space::WARMUP_STEPS_RANGE),
        range_bounds(candidate.start_ratio, candidate_space::START_RATIO_RANGE),
        range_bounds(candidate.amuse_beta1, candidate_space::AMUSE_BETA1_RANGE),
        range_bounds(candidate.amuse_rho, candidate_space::AMUSE_RHO_RANGE),
    ]
}

fn range(value: f64, min: f64, max: f64) -> f64 {
    if max <= min {
        return 0.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn log_range(value: f64, bounds: (f64, f64)) -> f64 {
    range(value.ln(), bounds.0.ln(), bounds.1.ln())
}

fn range_bounds(value: f64, bounds: (f64, f64)) -> f64 {
    range(value, bounds.0, bounds.1)
}

fn range_usize(value: usize, bounds: (usize, usize)) -> f64 {
    range(value as f64, bounds.0 as f64, bounds.1 as f64)
}
