//! Complete composition checks declared before production integration.
use super::{cli::TrainArgs, output::build_run_info};
use crate::AppResult;
use clap::Parser;

fn args(budget: Option<usize>) -> Vec<String> {
    let mut args = [
        "rust-kernels",
        "--bounded-swiglu-engram-mhc",
        "--full-kda-backward",
        "--steps",
        "20000",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if let Some(budget) = budget {
        args.extend(["--engram-table-params".into(), budget.to_string()]);
    }
    args
}

#[test]
fn complete_bounded_cli_preserves_engram_mhc_full_credit_and_step_target() {
    let parsed = TrainArgs::try_parse_from(args(None))
        .expect("complete bounded graph flag must be supported");
    let config = parsed.config();
    assert!(parsed.full_kda_backward);
    assert!(config.engram_mhc && config.mhc && !config.latent_feedback);
    assert_eq!(config.step_cap, 20000);
}

#[test]
fn complete_bounded_cli_preserves_explicit_table_budget() {
    let parsed = TrainArgs::try_parse_from(args(Some(500_000_000)))
        .expect("complete bounded graph must retain its lookup capacity");
    assert_eq!(parsed.config().engram_table_params, Some(500_000_000));
}

#[test]
fn complete_bounded_cli_rejects_missing_credit_and_other_graphs() {
    assert!(TrainArgs::try_parse_from(["rust-kernels", "--bounded-swiglu-engram-mhc"]).is_err());
    for other in [
        "--engram-mhc",
        "--mhc",
        "--latent-feedback",
        "--bounded-swiglu",
        "--value-rms",
        "--stationary-rope",
    ] {
        let mut argv = args(None);
        argv.push(other.into());
        assert!(TrainArgs::try_parse_from(argv).is_err(), "other={other}");
    }
}

#[test]
fn matching_bounded_build_reports_all_active_components_and_input_precision() {
    if !(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD) {
        return;
    }
    let parsed = TrainArgs::try_parse_from(args(Some(500_000_000))).unwrap();
    let info = build_run_info("fineweb", &parsed.config());
    for expected in [
        "mlp_kind=bounded-swiglu",
        "mlp_activation_input_bits=16",
        "engram_enabled=true",
        "engram_mhc_enabled=true",
        "mhc_enabled=true",
        "KDA_backward=full_recurrent_state_adjoint",
        "engram_table_parameters_requested=500000000",
    ] {
        assert!(
            info.lines().any(|line| line == expected),
            "missing {expected}"
        );
    }
}

#[test]
#[ignore = "actual complete B8 bounded-SwiGLU/Engram/mHC/full-credit launcher, both table budgets; all gradients, replays and causality"]
fn native_complete_bounded_engram_launcher_gradients_replay_and_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    for budget in [None, Some(500_000_000)] {
        let config = TrainArgs::try_parse_from(args(budget))?.config();
        let trainer = config.create_trainer()?;
        super::super::engram::multi_stream_tests::verify_complete_trainer(trainer, budget)?;
    }
    Ok(())
}
