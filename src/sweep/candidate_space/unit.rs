use super::{
    AMUSE_BETA1_RANGE, AMUSE_RHO_RANGE, BATCH_SIZE, FACTOR_COUNT, LR_SCALE_RANGE, N_EMBD, N_LAYER,
    START_RATIO_RANGE, WARMUP_STEPS_RANGE, valid_aurora_blocks, valid_aurora_phases,
};
use crate::sweep::candidate::Candidate;

pub(in crate::sweep) fn from_unit(unit: [f64; FACTOR_COUNT]) -> Candidate {
    let (n_embd, n_head) = choose_unit(&N_EMBD, unit[2]);
    let n_layer = choose_unit(&N_LAYER, unit[1]);
    let blocks = valid_aurora_blocks(n_layer);
    let aurora_blocks = choose_unit(&blocks, unit[4]);
    let phases = valid_aurora_phases(n_layer * 4, aurora_blocks);
    Candidate {
        batch_size: choose_unit(&BATCH_SIZE, unit[0]),
        n_layer,
        n_embd,
        n_head,
        aurora_phases: choose_unit(&phases, unit[3]),
        aurora_blocks,
        lr_scale: log_lerp(LR_SCALE_RANGE, unit[5]),
        adam_lr_scale: log_lerp(LR_SCALE_RANGE, unit[6]),
        nextlat_lr_scale: log_lerp(LR_SCALE_RANGE, unit[7]),
        warmup_steps: range_usize(WARMUP_STEPS_RANGE, unit[8]),
        start_ratio: range_f64(START_RATIO_RANGE, unit[9]),
        amuse_beta1: range_f64(AMUSE_BETA1_RANGE, unit[10]),
        amuse_rho: range_f64(AMUSE_RHO_RANGE, unit[11]),
    }
}

pub(in crate::sweep) fn choose_unit<T: Copy>(values: &[T], unit: f64) -> T {
    let index = (unit.clamp(0.0, 1.0) * values.len() as f64).floor() as usize;
    values[index.min(values.len() - 1)]
}

pub(in crate::sweep) fn log_lerp(range: (f64, f64), unit: f64) -> f64 {
    let lo = range.0.ln();
    let hi = range.1.ln();
    (lo + (hi - lo) * unit.clamp(0.0, 1.0)).exp()
}

pub(in crate::sweep) fn range_f64(range: (f64, f64), unit: f64) -> f64 {
    range.0 + (range.1 - range.0) * unit.clamp(0.0, 1.0)
}

pub(in crate::sweep) fn range_usize(range: (usize, usize), unit: f64) -> usize {
    let span = (range.1 - range.0) as f64;
    range.0 + (span * unit.clamp(0.0, 1.0)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::{FACTOR_COUNT, from_unit, valid_aurora_phases};

    #[test]
    fn unit_mapping_keeps_candidate_in_valid_space() {
        let low = from_unit([0.0; FACTOR_COUNT]);
        let high = from_unit([1.0; FACTOR_COUNT]);

        assert_eq!(low.batch_size, 1);
        assert_eq!(low.n_layer, 16);
        assert_eq!(low.n_embd, 2048);
        assert_eq!(low.n_head, 32);
        assert_eq!(high.batch_size, 32);
        assert_eq!(high.n_layer, 16);
        assert_eq!(high.n_embd, 2048);
        assert_eq!(high.n_head, 32);
        assert!(
            valid_aurora_phases(low.n_layer * 4, low.aurora_blocks).contains(&low.aurora_phases)
        );
        assert!(
            valid_aurora_phases(high.n_layer * 4, high.aurora_blocks).contains(&high.aurora_phases)
        );
    }
}
