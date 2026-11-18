//! Complete existing position/Engram/mHC failure cases declared before integration.
use super::{cli::TrainArgs, output::build_run_info};
use crate::AppResult;
use clap::Parser;

fn args(bounded: bool, value: bool, budget: Option<usize>) -> Vec<String> {
    let mut argv = [
        "rust-kernels",
        "--stationary-rope-engram-mhc",
        "--full-kda-backward",
        "--steps",
        "20000",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if bounded {
        argv.push("--bounded-swiglu-engram-mhc".into());
    }
    if value {
        argv.push("--value-rms-engram-mhc".into());
    }
    if let Some(budget) = budget {
        argv.extend(["--engram-table-params".into(), budget.to_string()]);
    }
    argv
}

#[test]
fn complete_stationary_cli_retains_all_selected_components_and_both_budgets() {
    for bounded in [false, true] {
        for value in [false, true] {
            for budget in [None, Some(500_000_000)] {
                let parsed = TrainArgs::try_parse_from(args(bounded, value, budget))
                    .expect("complete positional selector must exist and compose");
                let config = parsed.config();
                assert!(parsed.full_kda_backward);
                assert_eq!(parsed.bounded_swiglu_engram_mhc, bounded);
                assert_eq!(parsed.value_rms_engram_mhc, value);
                assert!(config.engram_mhc && config.mhc && !config.latent_feedback);
                assert_eq!(config.engram_table_params, budget);
                assert_eq!(config.step_cap, 20000);
            }
        }
    }
}

#[test]
fn complete_stationary_cli_rejects_missing_credit_and_standalone_graph_mixtures() {
    assert!(TrainArgs::try_parse_from(["rust-kernels", "--stationary-rope-engram-mhc"]).is_err());
    for other in [
        "--engram-mhc",
        "--mhc",
        "--latent-feedback",
        "--stationary-rope",
        "--bounded-swiglu",
        "--value-rms",
    ] {
        let mut argv = args(true, true, None);
        argv.push(other.into());
        assert!(TrainArgs::try_parse_from(argv).is_err(), "conflict={other}");
    }
}

#[test]
fn matching_complete_stationary_build_reports_the_real_graph_and_original_map() {
    if !(gpt2_nvfp4::GPT2_STATIONARY_ROPE && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD) {
        return;
    }
    let parsed = TrainArgs::try_parse_from(args(
        gpt2_nvfp4::GPT2_BOUNDED_SWIGLU,
        gpt2_nvfp4::GPT2_VALUE_RMS,
        Some(500_000_000),
    ))
    .unwrap();
    let info = build_run_info("fineweb", &parsed.config());
    for expected in [
        "stationary_rope_enabled=true",
        "position_stationary_dimensions=32..64",
        "position_frequency_base=1024",
        "position_rotation_sign=negative",
        "position_key_offset=stationary_half_only",
        "position_trigonometry=native_FP32",
        "engram_enabled=true",
        "engram_mhc_enabled=true",
        "mhc_enabled=true",
        "KDA_backward=full_recurrent_state_adjoint",
    ] {
        assert!(
            info.lines().any(|line| line == expected),
            "missing {expected}"
        );
    }
}

#[test]
#[ignore = "actual complete B8 stationary-position/Engram/mHC/full-credit factory; selected full MLP/value graph, both budgets, original gradient/replay/causality gates"]
fn native_complete_stationary_engram_launcher_gradients_replay_and_causality() -> AppResult {
    assert!(gpt2_nvfp4::GPT2_STATIONARY_ROPE && gpt2_nvfp4::GPT2_KDA_FULL_STATE_BACKWARD);
    for budget in [None, Some(500_000_000)] {
        let config = TrainArgs::try_parse_from(args(
            gpt2_nvfp4::GPT2_BOUNDED_SWIGLU,
            gpt2_nvfp4::GPT2_VALUE_RMS,
            budget,
        ))?
        .config();
        super::super::engram::multi_stream_tests::verify_complete_trainer(
            config.create_trainer()?,
            budget,
        )?;
    }
    Ok(())
}
