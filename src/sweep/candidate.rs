use super::{candidate_space, fmt, rng::SweepRng};

pub const MIN_N_LAYER: usize = 16;
pub const MIN_N_EMBD: usize = 2048;
pub const MIN_N_HEAD: usize = 32;
pub(super) use candidate_space::valid_muon_phases;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub batch_size: usize,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub muon_phases: usize,
    pub muon_blocks: usize,
    pub lr_scale: f64,
    pub adam_lr_scale: f64,
    pub nextlat_lr_scale: f64,
    pub warmup_steps: usize,
    pub start_ratio: f64,
    pub amuse_beta1: f64,
    pub amuse_rho: f64,
}

impl Candidate {
    pub fn random(rng: &mut SweepRng) -> Self {
        candidate_space::random(rng)
    }

    pub fn meets_model_floor(&self) -> bool {
        self.n_layer >= MIN_N_LAYER && self.n_embd >= MIN_N_EMBD && self.n_head >= MIN_N_HEAD
    }

    pub fn with_model_floor(&self) -> Self {
        if self.meets_model_floor() {
            return self.clone();
        }

        let n_layer = self.n_layer.max(MIN_N_LAYER);
        let n_embd = self.n_embd.max(MIN_N_EMBD);
        let n_head = self.n_head.max(MIN_N_HEAD);
        let phases = candidate_space::valid_muon_phases(n_layer * 4, self.muon_blocks);
        let muon_phases = phases
            .iter()
            .copied()
            .find(|phase| *phase >= self.muon_phases)
            .or_else(|| phases.first().copied())
            .unwrap_or(self.muon_phases);
        Self {
            n_layer,
            n_embd,
            n_head,
            muon_phases,
            ..self.clone()
        }
    }

    pub fn key(&self) -> String {
        format!(
            "{}_lr{:.4}_alr{:.4}_nlr{:.4}_w{}_s{:.2}_b{:.2}_r{:.2}",
            self.build_key(),
            self.lr_scale,
            self.adam_lr_scale,
            self.nextlat_lr_scale,
            self.warmup_steps,
            self.start_ratio,
            self.amuse_beta1,
            self.amuse_rho
        )
    }

    pub fn build_key(&self) -> String {
        format!(
            "b{}_l{}_d{}_h{}_p{}_c{}",
            self.batch_size,
            self.n_layer,
            self.n_embd,
            self.n_head,
            self.muon_phases,
            self.muon_blocks,
        )
    }

    pub fn build_env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("GPT2_BATCH_SIZE", self.batch_size.to_string()),
            ("GPT2_N_LAYER", self.n_layer.to_string()),
            ("GPT2_N_EMBD", self.n_embd.to_string()),
            ("GPT2_N_HEAD", self.n_head.to_string()),
            ("MUON_MATRIX_PHASES", self.muon_phases.to_string()),
            ("MUON_COOPERATIVE_BLOCKS", self.muon_blocks.to_string()),
        ]
    }

    pub fn run_env(&self) -> Vec<(&'static str, String)> {
        vec![
            ("TRAIN_LR_SCALE", fmt::f64_6(self.lr_scale)),
            ("TRAIN_ADAM_LR_SCALE", fmt::f64_6(self.adam_lr_scale)),
            ("TRAIN_NEXTLAT_LR_SCALE", fmt::f64_6(self.nextlat_lr_scale)),
            ("TRAIN_LR_WARMUP_STEPS", self.warmup_steps.to_string()),
            ("TRAIN_LR_START_RATIO", fmt::f64_6(self.start_ratio)),
            ("TRAIN_AMUSE_BETA1", fmt::f64_6(self.amuse_beta1)),
            ("TRAIN_AMUSE_RHO", fmt::f64_6(self.amuse_rho)),
        ]
    }
}
