use super::super::{
    analysis::{self, SweepAnalysis},
    config::SweepConfig,
};

#[derive(Default)]
pub struct Direction {
    pub batch_size: f64,
    pub n_layer: f64,
    pub n_embd: f64,
    pub muon_phases: f64,
    pub muon_blocks: f64,
    pub lr_scale: f64,
    pub adam_lr_scale: f64,
    pub nextlat_lr_scale: f64,
    pub warmup_steps: f64,
    pub start_ratio: f64,
    pub amuse_beta1: f64,
    pub amuse_rho: f64,
}

pub fn from_analysis(analysis: &SweepAnalysis, config: &SweepConfig) -> Direction {
    let mut direction = Direction::default();
    for belief in analysis::factor_beliefs(analysis, config) {
        add(&mut direction, &belief.factor, belief.direction);
    }
    direction
}

fn add(direction: &mut Direction, name: &str, value: f64) {
    match name {
        "batch_size" => direction.batch_size += value,
        "n_layer" => direction.n_layer += value,
        "n_embd" => direction.n_embd += value,
        "muon_phases" => direction.muon_phases += value,
        "muon_blocks" => direction.muon_blocks += value,
        "ln_lr_scale" => direction.lr_scale += value,
        "ln_adam_lr_scale" => direction.adam_lr_scale += value,
        "ln_nextlat_lr_scale" => direction.nextlat_lr_scale += value,
        "ln_warmup_steps" => direction.warmup_steps += value,
        "start_ratio" => direction.start_ratio += value,
        "amuse_beta1" => direction.amuse_beta1 += value,
        "amuse_rho" => direction.amuse_rho += value,
        _ => {}
    }
}
