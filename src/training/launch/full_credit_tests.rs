//! Declared before allowing full recurrent credit on the working combined graph.
use super::{cli::TrainArgs, config::TrainConfig, output::build_run_info};
use clap::Parser;

#[test]
fn full_credit_combined_cli_keeps_the_complete_graph() {
    let args = TrainArgs::try_parse_from([
        "rust-kernels",
        "--engram-mhc",
        "--full-kda-backward",
        "--steps",
        "20000",
    ])
    .expect("full recurrent credit must compose with the complete working Engram/mHC graph");
    assert!(args.full_kda_backward);
    let config = args.config();
    assert!(config.engram_mhc && config.mhc && !config.latent_feedback);
    assert_eq!(config.step_cap, 20000);
}

#[test]
fn full_credit_combined_cli_retains_explicit_table_capacity() {
    let args = TrainArgs::try_parse_from([
        "rust-kernels",
        "--engram-mhc",
        "--full-kda-backward",
        "--engram-table-params",
        "500000000",
    ])
    .expect("backward semantics must not drop the declared complete table configuration");
    let config = args.config();
    assert!(config.engram_mhc && config.mhc);
    assert_eq!(config.engram_table_params, Some(500_000_000));
}

#[test]
fn full_credit_combined_report_keeps_memory_active() {
    let mut config = TrainConfig::from_env();
    config.engram_mhc = true;
    config.mhc = true;
    config.latent_feedback = false;
    let info = build_run_info("fineweb", &config);
    assert!(info.lines().any(|line| line == "engram_enabled=true"));
    assert!(info.lines().any(|line| line == "engram_mhc_enabled=true"));
    if gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD {
        assert!(
            info.lines()
                .any(|line| line == "KDA_backward=full_recurrent_state_adjoint")
        );
    }
}

#[test]
fn full_credit_keeps_other_unverified_graph_combinations_rejected() {
    for other in [
        "--mhc",
        "--latent-feedback",
        "--bounded-swiglu",
        "--value-rms",
        "--stationary-rope",
    ] {
        assert!(
            TrainArgs::try_parse_from(["rust-kernels", "--full-kda-backward", other]).is_err(),
            "other={other}"
        );
    }
}
