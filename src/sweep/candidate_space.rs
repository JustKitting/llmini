mod sample;
mod unit;

use super::candidate::{MIN_N_EMBD, MIN_N_HEAD, MIN_N_LAYER};

pub(in crate::sweep) use sample::{random, valid_muon_blocks, valid_muon_phases};
pub(in crate::sweep) use unit::{choose_unit, from_unit, log_lerp, range_f64, range_usize};

pub const BATCH_SIZE: [usize; 8] = [1, 2, 4, 8, 12, 16, 24, 32];
pub const N_LAYER: [usize; 1] = [MIN_N_LAYER];
pub const N_EMBD: [(usize, usize); 1] = [(MIN_N_EMBD, MIN_N_HEAD)];
pub const MUON_BLOCKS: [usize; 5] = [80, 90, 120, 160, 180];
pub const LR_SCALE_RANGE: (f64, f64) = (0.5, 2.5);
pub const WARMUP_STEPS_RANGE: (usize, usize) = (5, 100);
pub const START_RATIO_RANGE: (f64, f64) = (0.0, 0.2);
pub const AMUSE_BETA1_RANGE: (f64, f64) = (0.2, 0.6);
pub const AMUSE_RHO_RANGE: (f64, f64) = (0.5, 1.0);
pub const FACTOR_COUNT: usize = 12;
