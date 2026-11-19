//! CLI failure cases declared before inspector implementation.
use super::cli::TrainArgs;
use clap::Parser;
#[test]
fn decay_inspection_requires_model_inspection_without_training_actions() {
    assert!(
        TrainArgs::try_parse_from([
            "rust-kernels",
            "--inspect-model",
            "model.ckpt",
            "--inspect-kda-decay"
        ])
        .is_ok()
    );
    for args in [
        vec!["rust-kernels", "--inspect-kda-decay"],
        vec!["rust-kernels", "--inspect-kda-decay", "--mhc"],
        vec![
            "rust-kernels",
            "--inspect-kda-decay",
            "--inspect-gradients",
            "model.ckpt",
        ],
        vec![
            "rust-kernels",
            "--inspect-model",
            "model.ckpt",
            "--inspect-kda-decay",
            "--steps",
            "1",
        ],
    ] {
        assert!(TrainArgs::try_parse_from(&args).is_err(), "args={args:?}");
    }
}
