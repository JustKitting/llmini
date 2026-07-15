use super::super::{
    candidate::{Candidate, valid_aurora_phases},
    candidate_space::{
        self, AMUSE_BETA1_RANGE, AMUSE_RHO_RANGE, BATCH_SIZE, LR_SCALE_RANGE, N_EMBD, N_LAYER,
        START_RATIO_RANGE, WARMUP_STEPS_RANGE,
    },
    rng::SweepRng,
};
use super::direction::Direction;

pub fn candidate(rng: &mut SweepRng, direction: &Direction, jitter: bool) -> Candidate {
    let batch_size = pick(&BATCH_SIZE, direction.batch_size, rng, jitter);
    let n_layer = pick(&N_LAYER, direction.n_layer, rng, jitter);
    let (n_embd, n_head) = pick(&N_EMBD, direction.n_embd, rng, jitter);
    let blocks = candidate_space::valid_aurora_blocks(n_layer);
    let aurora_blocks = pick(&blocks, direction.aurora_blocks, rng, jitter);
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size,
        n_layer,
        n_embd,
        n_head,
        aurora_phases: pick(&phases, direction.aurora_phases, rng, jitter),
        aurora_blocks,
        lr_scale: pick_f64(LR_SCALE_RANGE, direction.lr_scale, rng, jitter, true),
        adam_lr_scale: pick_f64(LR_SCALE_RANGE, direction.adam_lr_scale, rng, jitter, true),
        nextlat_lr_scale: pick_f64(
            LR_SCALE_RANGE,
            direction.nextlat_lr_scale,
            rng,
            jitter,
            true,
        ),
        warmup_steps: pick_usize(WARMUP_STEPS_RANGE, direction.warmup_steps, rng, jitter),
        start_ratio: pick_f64(START_RATIO_RANGE, direction.start_ratio, rng, jitter, false),
        amuse_beta1: pick_f64(AMUSE_BETA1_RANGE, direction.amuse_beta1, rng, jitter, false),
        amuse_rho: pick_f64(AMUSE_RHO_RANGE, direction.amuse_rho, rng, jitter, false),
    }
}

fn pick<T: Copy>(values: &[T], direction: f64, rng: &mut SweepRng, jitter: bool) -> T {
    if jitter && rng.f64() < 0.25 {
        return rng.choose(values);
    }
    candidate_space::choose_unit(values, directed_unit(direction))
}

fn pick_f64(
    range: (f64, f64),
    direction: f64,
    rng: &mut SweepRng,
    jitter: bool,
    log_scale: bool,
) -> f64 {
    let t = pick_unit(direction, rng, jitter);
    if log_scale {
        return candidate_space::log_lerp(range, t);
    }
    candidate_space::range_f64(range, t)
}

fn pick_usize(range: (usize, usize), direction: f64, rng: &mut SweepRng, jitter: bool) -> usize {
    candidate_space::range_usize(range, pick_unit(direction, rng, jitter))
}

fn pick_unit(direction: f64, rng: &mut SweepRng, jitter: bool) -> f64 {
    if jitter && rng.f64() < 0.25 {
        return rng.f64();
    }

    let center = directed_unit(direction);
    if jitter {
        return (center + (rng.f64() - 0.5) * 0.25).clamp(0.0, 1.0);
    }
    center
}

fn directed_unit(direction: f64) -> f64 {
    (0.5 + direction.tanh() * 0.35).clamp(0.0, 1.0 - f64::EPSILON)
}
