//! CLI/report failure cases declared before capacity-option implementation.
use super::{cli::TrainArgs, output::build_run_info};
use clap::Parser;
#[test]
fn explicit_table_budget_reaches_native_configuration_and_report() {
    let args = TrainArgs::try_parse_from([
        "rust-kernels",
        "--engram-mhc",
        "--engram-table-params",
        "500000000",
    ])
    .unwrap();
    let config = args.config();
    assert!(config.engram_mhc);
    assert_eq!(config.engram_table_params, Some(500_000_000));
    let info = build_run_info("fineweb", &config);
    assert!(
        info.lines()
            .any(|s| s == "engram_table_parameters_requested=500000000")
    );
    let actual = info
        .lines()
        .find_map(|s| s.strip_prefix("engram_table_parameters="))
        .unwrap()
        .parse::<usize>()
        .unwrap();
    assert!((500_000_000..501_000_000).contains(&actual));
    println!("configured_engram_table_budget requested=500000000 actual={actual}");
}
#[test]
fn table_budget_requires_the_complete_graph_and_positive_input() {
    for args in [
        vec!["rust-kernels", "--engram-table-params", "500000000"],
        vec!["rust-kernels", "--engram-mhc", "--engram-table-params", "0"],
        vec![
            "rust-kernels",
            "--mhc",
            "--engram-table-params",
            "500000000",
        ],
    ] {
        assert!(TrainArgs::try_parse_from(&args).is_err(), "args={args:?}");
    }
}

#[test]
fn four_gram_launch_selects_the_complete_graph_and_reports_its_shape() {
    assert!(TrainArgs::try_parse_from(["rust-kernels", "--engram-four-grams"]).is_err());
    let config = TrainArgs::try_parse_from([
        "rust-kernels",
        "--stationary-rope-engram-mhc",
        "--value-rms-engram-mhc",
        "--full-kda-backward",
        "--engram-four-grams",
        "--engram-table-params",
        "500000000",
    ])
    .unwrap()
    .config();
    assert!(config.engram_mhc && config.engram_four_grams);
    assert_eq!(config.engram_table_params, Some(500_000_000));
    let info = build_run_info("fineweb", &config);
    for expected in [
        "engram_checkpoint_graph_version=5",
        "engram_memory_width=1536",
        "engram_ngram_orders=2,3,4",
    ] {
        assert!(info.lines().any(|line| line == expected), "{expected}");
    }
}
