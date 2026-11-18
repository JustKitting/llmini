use super::cli::TrainArgs;
use clap::Parser;
#[test]
fn stationary_rope_cli_keeps_graph_trials_separate() {
    assert!(TrainArgs::try_parse_from(["rust-kernels", "--stationary-rope"]).is_ok());
    for other in [
        "--engram-mhc",
        "--mhc",
        "--latent-feedback",
        "--bounded-swiglu",
        "--value-rms",
        "--full-kda-backward",
    ] {
        assert!(
            TrainArgs::try_parse_from(["rust-kernels", "--stationary-rope", other]).is_err(),
            "conflict={other}"
        );
    }
}
