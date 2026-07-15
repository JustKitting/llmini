use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run coupled validation-loss sweeps for rust-kernels training")]
pub struct SweepConfig {
    #[arg(long, default_value_t = 12)]
    pub trials: usize,
    #[arg(long, default_value_t = 4)]
    pub random_trials: usize,
    #[arg(long, default_value_t = 128)]
    pub candidate_samples: usize,
    #[arg(long, default_value_t = 900.0)]
    pub max_seconds: f64,
    #[arg(long, default_value_t = 30.0)]
    pub screen_max_seconds: f64,
    #[arg(long, default_value_t = 1.0)]
    pub sweep_quality_weight: f64,
    #[arg(long, default_value_t = 0.75)]
    pub sweep_stability_weight: f64,
    #[arg(long, default_value_t = 0.35)]
    pub sweep_exploration_weight: f64,
    #[arg(long, default_value_t = 1)]
    pub log_interval: usize,
    #[arg(long, default_value = "fineweb")]
    pub dataset: String,
    #[arg(long, default_value = "sm_120a")]
    pub arch: String,
    #[arg(long)]
    pub cuda_device: Option<String>,
    #[arg(long)]
    pub sweep_dir: Option<PathBuf>,
    #[arg(long, default_value = "notes/sweep_seed_nextlat.tsv")]
    pub seed_history: PathBuf,
    #[arg(long, default_value = "notes/sweep_baseline.env")]
    pub baseline: PathBuf,
    #[arg(long, default_value_t = 0x4750_5432)]
    pub seed: u64,
    #[arg(long)]
    pub dry_run: bool,
}
