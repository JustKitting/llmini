//! Host-side Muon update flow.

mod groups;
mod tma;

use gpt2_nvfp4::{GPT2_MLP, GPT2_N_EMBD, GPT2_N_LAYER, GPT2_QKV, NEXTLAT_HIDDEN, NEXTLAT_INPUT};

pub(super) use groups::{MuonGroupTable, MuonPointerTables};
pub(super) use tma::{MuonTmaArgs, apply_muon_tma};

const SIGN_MUON_BETA: f32 = 0.9;
const SIGN_MUON_PERIOD: u32 = 2;
const SIGN_MUON_FULL_LR_MULTIPLIER: f32 = 2.0;
const SIGN_MUON_SIGN_LR_RATIO: f32 = 0.025;
const POLAR_ITERATIONS: u32 = 5;
pub(super) const MUON_LR: f32 = 1.0e-4;
pub(super) const MUON_WEIGHT_DECAY: f32 = 0.025;
pub(in crate::training) const MUON_MATRIX_SLOTS: usize = GPT2_N_LAYER * 4 + 3;

pub(super) fn muon_learning_rate(step: u32) -> f32 {
    let update_ratio = if sign_muon_uses_polar(step) {
        1.0
    } else {
        SIGN_MUON_SIGN_LR_RATIO
    };
    MUON_LR
        * SIGN_MUON_FULL_LR_MULTIPLIER
        * update_ratio
        * super::learning_rate::muon_multiplier(step)
}

fn sign_muon_uses_polar(step: u32) -> bool {
    step.saturating_sub(1).is_multiple_of(SIGN_MUON_PERIOD)
}

pub(in crate::training) const fn max_matrix_len() -> usize {
    max3(
        GPT2_N_EMBD * GPT2_QKV,
        GPT2_MLP * GPT2_N_EMBD,
        NEXTLAT_INPUT * NEXTLAT_HIDDEN,
    )
}

pub(in crate::training) const fn max_matrix_dim() -> usize {
    max2(GPT2_N_EMBD, NEXTLAT_HIDDEN)
}

pub(in crate::training) const fn max_polar_cols() -> usize {
    max3(max2(GPT2_QKV, GPT2_MLP), NEXTLAT_INPUT, NEXTLAT_HIDDEN)
}

const fn max2(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn max3(a: usize, b: usize, c: usize) -> usize {
    max2(max2(a, b), c)
}

#[cfg(test)]
mod tests {
    use super::sign_muon_uses_polar;

    #[test]
    fn sign_muon_period_starts_with_polar_and_alternates() {
        assert!(sign_muon_uses_polar(1));
        assert!(!sign_muon_uses_polar(2));
        assert!(sign_muon_uses_polar(3));
        assert!(!sign_muon_uses_polar(4));
    }
}
