//! Complete value-RMS composition checks declared before production integration.
use super::{cli::TrainArgs, output::build_run_info};
use crate::AppResult;
use clap::Parser;

fn args(bounded: bool, budget: Option<usize>) -> Vec<String> {
    let mut args = [
        "rust-kernels",
        "--value-rms-engram-mhc",
        "--full-kda-backward",
        "--steps",
        "20000",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if bounded {
        args.push("--bounded-swiglu-engram-mhc".into());
    }
    if let Some(budget) = budget {
        args.extend(["--engram-table-params".into(), budget.to_string()]);
    }
    args
}

#[test]
fn complete_value_cli_retains_the_actual_graph_and_both_table_budgets() {
    for budget in [None, Some(500_000_000)] {
        let parsed = TrainArgs::try_parse_from(args(false, budget))
            .expect("complete value-RMS selector must exist");
        let config = parsed.config();
        assert!(parsed.full_kda_backward);
        assert!(config.engram_mhc && config.mhc && !config.latent_feedback);
        assert_eq!(config.engram_table_params, budget);
        assert_eq!(config.step_cap, 20000);
    }
}

#[test]
fn complete_value_cli_can_retain_the_existing_bounded_MLP() {
    let parsed = TrainArgs::try_parse_from(args(true, Some(500_000_000)))
        .expect("complete value-RMS and bounded-SwiGLU selectors must compose");
    assert!(parsed.bounded_swiglu_engram_mhc && parsed.full_kda_backward);
    assert!(parsed.config().engram_mhc && parsed.config().mhc);
}

#[test]
fn complete_value_cli_rejects_missing_credit_and_unsupported_graphs() {
    assert!(TrainArgs::try_parse_from(["rust-kernels", "--value-rms-engram-mhc"]).is_err());
    for bounded in [false, true] {
        for other in [
            "--engram-mhc",
            "--mhc",
            "--value-rms",
            "--bounded-swiglu",
            "--stationary-rope",
        ] {
            let mut argv = args(bounded, None);
            argv.push(other.into());
            assert!(
                TrainArgs::try_parse_from(argv).is_err(),
                "bounded={bounded} other={other}"
            );
        }
    }
}

#[test]
fn matching_complete_value_build_reports_all_components_and_represented_precision() {
    if !(gpt2_nvfp4::GPT2_VALUE_RMS && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD) {
        return;
    }
    let parsed =
        TrainArgs::try_parse_from(args(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU, Some(500_000_000)))
            .unwrap();
    let info = build_run_info("fineweb", &parsed.config());
    for expected in [
        "value_rms_enabled=true",
        "value_rms_precision_policy_version=2",
        "value_rms_reverse_SiLU_input_bits=16",
        "value_rms_reverse_normalized_direction_bits=16",
        "value_rms_normalization_control_bits=32",
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
#[ignore = "actual complete B8 value-RMS/Engram/mHC/full-credit factory, both budgets and selected full MLP; all gradients, tapes, replay and causality"]
fn native_complete_value_engram_launcher_gradients_replay_and_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_VALUE_RMS && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    for budget in [None, Some(500_000_000)] {
        let config =
            TrainArgs::try_parse_from(args(gpt2_nvfp4::GPT2_BOUNDED_SWIGLU, budget))?.config();
        let trainer = config.create_trainer()?;
        super::super::engram::multi_stream_tests::verify_complete_trainer(trainer, budget)?;
    }
    Ok(())
}
