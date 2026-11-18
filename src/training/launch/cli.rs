use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;

use clap::Parser;

use super::config::TrainConfig;

/// Train the native NVFP4 CUDA language model.
#[derive(Debug, Parser)]
#[command(version)]
#[command(group(clap::ArgGroup::new("checkpoint_action").args(["evaluate_model","inspect_model","inspect_gradients","inspect_feedback_depths"]).multiple(false)))]
#[command(group(clap::ArgGroup::new("evaluation_input").args(["validation_documents","validation_answers"]).multiple(false)))]
#[command(group(clap::ArgGroup::new("complete_engram").args(["engram_mhc","bounded_swiglu_engram_mhc","value_rms_engram_mhc","stationary_rope_engram_mhc"]).multiple(true)))]
pub(super) struct TrainArgs {
    /// Hash-bound native replay record for a verified numerical graph repair.
    #[arg(long, requires = "checkpoint_action")]
    pub(super) replay_provenance: Option<PathBuf>,
    /// Training corpus; falls back to TRAIN_DATASET when omitted.
    #[arg(long, value_parser = ["fineweb", "synth", "shakespeare"])]
    pub(super) dataset: Option<String>,

    /// Read a JSON manifest with a tokenizer name and ordered native u16 training shards.
    #[arg(long, conflicts_with_all = ["dataset", "checkpoint_action"])]
    pub(super) training_shards: Option<PathBuf>,

    /// Train the complete latent-feedback architecture (replaces the Engram experiment).
    #[arg(long, conflicts_with_all=["inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    latent_feedback: bool,

    /// Train four-stream mHC with independent attention/MLP mixers and a learned HyperHead.
    #[arg(long, conflicts_with_all=["latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    mhc: bool,
    /// Train complete four-stream Engram with32 mHC boundaries and learned HyperHead.
    #[arg(long, conflicts_with_all=["mhc","latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    engram_mhc: bool,

    /// Train complete Engram/mHC with bounded SwiGLU and full recurrent KDA credit; matching build required.
    #[arg(long, requires="full_kda_backward", conflicts_with_all=["engram_mhc","mhc","latent_feedback","bounded_swiglu","value_rms","stationary_rope","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) bounded_swiglu_engram_mhc: bool,

    /// Train complete Engram/mHC/full-credit value RMS; add the bounded composite selector to retain its matching MLP.
    #[arg(long, requires="full_kda_backward", conflicts_with_all=["engram_mhc","mhc","latent_feedback","bounded_swiglu","value_rms","stationary_rope","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) value_rms_engram_mhc: bool,

    /// Train complete stationary-position/Engram/mHC/full-credit; retain selected numerical components with their composite selectors.
    #[arg(long, requires="full_kda_backward", conflicts_with_all=["engram_mhc","mhc","latent_feedback","bounded_swiglu","value_rms","stationary_rope","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) stationary_rope_engram_mhc: bool,

    /// Minimum lookup-table values across both full Engram insertions; fresh training only.
    #[arg(long, requires = "complete_engram", conflicts_with_all = ["mhc", "latent_feedback", "bounded_swiglu", "value_rms", "stationary_rope", "inspect_model", "inspect_gradients", "evaluate_model", "inspect_feedback_depths"])]
    engram_table_params: Option<NonZeroUsize>,

    /// Train the separate complete bounded-SwiGLU graph; requires its matching native build.
    #[arg(long, conflicts_with_all=["engram_mhc","mhc","latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) bounded_swiglu: bool,

    /// Train the separate complete value-RMS graph; requires GPT2_VALUE_RMS=1.
    #[arg(long, conflicts_with_all=["bounded_swiglu","engram_mhc","mhc","latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) value_rms: bool,

    /// Train full recurrent KDA credit, optionally with --engram-mhc; matching native build required.
    #[arg(long, conflicts_with_all=["value_rms","bounded_swiglu","mhc","latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) full_kda_backward: bool,

    /// Train half-truncated base1024 RoPE with stationary-half key offset; matching build required.
    #[arg(long, conflicts_with_all=["full_kda_backward","value_rms","bounded_swiglu","engram_mhc","mhc","latent_feedback","inspect_model","inspect_gradients","evaluate_model","inspect_feedback_depths"])]
    pub(super) stationary_rope: bool,

    /// Maximum attempted training steps; overrides TRAIN_STEPS.
    #[arg(long)]
    steps: Option<NonZeroUsize>,

    /// Training-time ceiling in seconds; overrides TRAIN_MAX_SECONDS.
    #[arg(long, value_parser = positive_seconds)]
    max_seconds: Option<f64>,

    /// Evaluate every N completed training steps; overrides TRAIN_EVAL_INTERVAL.
    #[arg(long)]
    eval_interval: Option<NonZeroUsize>,

    /// Evaluate this saved native checkpoint without training.
    #[arg(long, requires = "evaluation_input",
        conflicts_with_all = ["dataset", "steps", "max_seconds", "eval_interval"])]
    pub(super) evaluate_model: Option<PathBuf>,

    /// Reserved reference documents produced by the native FineWeb preparer.
    #[arg(long, requires = "evaluate_model")]
    pub(super) validation_documents: Option<PathBuf>,

    /// Held-out JSONL records with task, prompt, and answer fields. Score the answer bytes.
    #[arg(long, requires = "evaluate_model")]
    pub(super) validation_answers: Option<PathBuf>,

    /// JSON destination; defaults to reference_validation.json beside the checkpoint.
    #[arg(long, requires = "evaluate_model")]
    pub(super) evaluation_output: Option<PathBuf>,

    /// Inspect CUDA activation tapes of a saved native checkpoint, without training.
    #[arg(long, conflicts_with_all = ["dataset", "steps", "max_seconds", "eval_interval",
        "evaluate_model", "validation_documents", "evaluation_output"])]
    pub(super) inspect_model: Option<PathBuf>,

    /// Count KDA decay-factor caps from exact native FP32 projection recomputation, without updates.
    #[arg(long,requires="inspect_model",conflicts_with_all=["evaluate_model","inspect_gradients","inspect_feedback_depths","mhc","engram_mhc","latent_feedback","bounded_swiglu","value_rms","full_kda_backward","stationary_rope","engram_table_params"])]
    pub(super) inspect_kda_decay: bool,

    /// Inspection JSON destination; defaults to activation_inspection.json beside the checkpoint.
    #[arg(long, requires = "inspect_model")]
    pub(super) inspection_output: Option<PathBuf>,

    /// Attribute the saved model's native CUDA gradients, without applying updates.
    #[arg(long, conflicts_with_all = ["dataset", "steps", "max_seconds", "eval_interval",
        "evaluate_model", "validation_documents", "evaluation_output", "inspect_model", "inspection_output"])]
    pub(super) inspect_gradients: Option<PathBuf>,

    /// JSON destination; defaults to gradient_inspection.json beside the checkpoint.
    #[arg(long, requires = "inspect_gradients")]
    pub(super) gradient_output: Option<PathBuf>,

    /// Inspect this native u16 corpus shard instead of the validation batch.
    #[arg(long, requires_all = ["inspect_gradients", "gradient_token_offset"])]
    pub(super) gradient_shard: Option<PathBuf>,

    /// First token of the corpus batch, using the compiled batch/context shape.
    #[arg(long, requires_all = ["inspect_gradients", "gradient_shard"])]
    pub(super) gradient_token_offset: Option<usize>,

    /// Use the base backward RNG draws from this 1-based training attempt.
    #[arg(long, requires = "inspect_gradients")]
    pub(super) gradient_rng_attempt: Option<NonZeroU32>,

    /// Compare held-out P1/P2/P3/P4 losses of a saved feedback model without training.
    #[arg(long, conflicts_with_all=["dataset","steps","max_seconds","eval_interval"])]
    pub(super) inspect_feedback_depths: Option<PathBuf>,

    /// JSON destination; defaults beside the checkpoint.
    #[arg(long, requires = "inspect_feedback_depths")]
    pub(super) feedback_depth_output: Option<PathBuf>,
}

impl TrainArgs {
    pub(super) fn config(&self) -> TrainConfig {
        let mut config = TrainConfig::from_env();
        if self.latent_feedback {
            config.latent_feedback = true;
        }
        if self.mhc {
            config.mhc = true;
        }
        if self.engram_mhc
            || self.bounded_swiglu_engram_mhc
            || self.value_rms_engram_mhc
            || self.stationary_rope_engram_mhc
            || config.engram_mhc
        {
            config.engram_mhc = true;
            config.mhc = true;
        }
        config.engram_table_params = self.engram_table_params.map(NonZeroUsize::get);
        if let Some(steps) = self.steps {
            config.step_cap = steps.get();
        }
        if let Some(seconds) = self.max_seconds {
            config.max_seconds = seconds;
        }
        if let Some(interval) = self.eval_interval {
            config.eval_interval = Some(interval.get());
        }
        config
    }
}

fn positive_seconds(value: &str) -> Result<f64, String> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| "expected a positive, finite number of seconds".to_owned())?;
    if seconds.is_finite() && seconds > 0.0 {
        Ok(seconds)
    } else {
        Err("expected a positive, finite number of seconds".to_owned())
    }
}

#[cfg(test)]
mod gradient_window_cli_tests {
    use super::*;

    #[test]
    fn gradient_inspector_accepts_the_observed_training_window() {
        let result = TrainArgs::try_parse_from([
            "rust-kernels",
            "--inspect-gradients",
            "model.ckpt",
            "--gradient-shard",
            "data/fineweb/shards/fineweb_mistral_v01_train_000007.bin",
            "--gradient-token-offset",
            "2129920",
            "--gradient-rng-attempt",
            "6231",
        ]);
        let args = result.expect("exact corpus-window probe was rejected");
        assert_eq!(args.gradient_rng_attempt.map(NonZeroU32::get), Some(6231));
    }

    #[test]
    fn corpus_window_arguments_are_paired_and_inspection_only() {
        for args in [
            vec![
                "rust-kernels",
                "--inspect-gradients",
                "model.ckpt",
                "--gradient-shard",
                "shard.bin",
            ],
            vec![
                "rust-kernels",
                "--inspect-gradients",
                "model.ckpt",
                "--gradient-token-offset",
                "0",
            ],
            vec![
                "rust-kernels",
                "--gradient-shard",
                "shard.bin",
                "--gradient-token-offset",
                "0",
            ],
            vec![
                "rust-kernels",
                "--inspect-gradients",
                "model.ckpt",
                "--gradient-shard",
                "shard.bin",
                "--gradient-token-offset",
                "-1",
            ],
            vec!["rust-kernels", "--gradient-rng-attempt", "6231"],
            vec![
                "rust-kernels",
                "--inspect-gradients",
                "model.ckpt",
                "--gradient-rng-attempt",
                "0",
            ],
        ] {
            assert!(TrainArgs::try_parse_from(args).is_err());
        }
    }
}
