use super::*;

// Declared before fixing the reached four-stream report failure. Match actual
// optimizer matrix names/indices; do not relabel key1 as the shared value.
#[test]
fn inspected_engram_projectors_follow_the_declared_graph() {
    for index in 0..2 {
        let one = super::super::engram::Config::reference(index);
        let four = super::super::engram::Config::reference_multistream(index);
        assert_eq!(
            inspected_engram_projectors(&one),
            vec![
                (format!("{}.key.weight", one.prefix()), 1),
                (format!("{}.value.weight", one.prefix()), 3)
            ]
        );
        let mut expected = (0..4)
            .map(|stream| {
                (
                    format!("{}.key.{stream}.weight", four.prefix()),
                    1 + 2 * stream,
                )
            })
            .collect::<Vec<_>>();
        expected.push((format!("{}.value.weight", four.prefix()), 9));
        assert_eq!(inspected_engram_projectors(&four), expected);
    }
}

#[test]
#[ignore = "exact captured four-stream profile checkpoint: all ten Engram and mHC matrix norm probes"]
fn native_four_stream_checkpoint_gradient_report_uses_actual_matrix_names() -> AppResult {
    let checkpoint = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/runs/20261116_023313Z_fineweb_150s/model.ckpt");
    let result = inspect_checkpoint(&checkpoint)?;
    assert_eq!(result["passed"], true);
    assert_eq!(result["optimizer_updates"], 0);
    assert_eq!(result["checkpoint_unchanged"], true);
    assert_eq!(result["ce_replay_bit_exact"], true);
    let checks = result["independent_matrix_checks"].as_array().unwrap();
    assert!(checks.iter().all(|v| v["passed"] == true));
    for layer in [1, 5] {
        for stream in 0..4 {
            assert!(
                checks
                    .iter()
                    .any(|v| v["name"] == format!("engram.layers.{layer}.key.{stream}.weight"))
            );
        }
        assert!(
            checks
                .iter()
                .any(|v| v["name"] == format!("engram.layers.{layer}.value.weight"))
        );
    }
    for name in [
        "mhc.boundaries.0.projection.weight",
        "mhc.head.projection.weight",
    ] {
        assert!(checks.iter().any(|v| v["name"] == name));
    }
    println!(
        "native_four_stream_gradient_report_verified independent_matrices={} model_unchanged=true updates=0",
        checks.len()
    );
    Ok(())
}

#[test]
#[ignore = "full native CUDA checkpoint gradient attribution; zero optimizer updates"]
fn native_checkpoint_gradient_components_reconcile_and_preserve_model() -> AppResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (run, expected_engram) in [
        ("20261114_080934Z_fineweb_21600s", false),
        ("20261115_002457Z_fineweb_43200s", true),
    ] {
        let checkpoint = root.join("target/runs").join(run).join("model.ckpt");
        let result = inspect_checkpoint(&checkpoint)?;
        assert_eq!(result["optimizer_updates"], 0);
        assert_eq!(result["optimizer_state_replayed"], false);
        assert_eq!(result["ce_replay_bit_exact"], true);
        assert_eq!(result["all_components_finite"], true);
        assert_eq!(result["global_norm_reconciles"], true);
        assert!(
            result["independent_matrix_checks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["passed"] == true)
        );
        assert_eq!(result["engram_present"], expected_engram);
        let components = result["components"].as_array().unwrap();
        assert!(components.len() > 200, "full model components required");
        let mut seen = std::collections::HashSet::new();
        let mut sum = 0.0;
        for component in components {
            let name = component["name"].as_str().unwrap();
            let kind = component["kind"].as_str().unwrap();
            assert!(seen.insert((name, kind)), "duplicate {name} {kind}");
            sum += component["squared_l2"].as_f64().unwrap();
        }
        let norm = result["global_gradient_norm"].as_f64().unwrap();
        assert!((sum.sqrt() - norm).abs() < 5e-5 * norm.max(1.0));
        assert!(seen.iter().any(|(_, kind)| *kind == "muon_matrix"));
        assert!(seen.iter().any(|(_, kind)| *kind == "symexp_control"));
        assert_eq!(
            seen.iter().any(|(name, _)| name.starts_with("engram.")),
            expected_engram
        );
        println!(
            "gradient_attribution_verified run={run} norm={norm} components={}",
            components.len()
        );
    }
    Ok(())
}

#[test]
#[ignore = "native full-model logging-cache verification; one fixture update, not loss evidence"]
fn native_live_components_preserve_preupdate_norm_after_optimizer() -> AppResult {
    let validation = TokenDataLoader::fineweb_validation_tokens()?;
    let mut trainer = Trainer::new(0x47505432)?;
    let windows = validation
        .chunks_exact(gpt2_nvfp4::GPT2_SEQ_LEN + 1)
        .cycle()
        .take(gpt2_nvfp4::GPT2_BATCH_SIZE)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let batch = trainer.batch_from_windows(&windows, gpt2_nvfp4::GPT2_BATCH_SIZE)?;
    let stats = trainer.train_step(&batch, true)?;
    assert!(!stats.optimizer.update_skipped);
    let snapshot = trainer.completed_gradient_components(1, stats.optimizer.grad_norm)?;
    assert_eq!(snapshot["passed"], true);
    assert_eq!(snapshot["scope"], "actual_live_training_update");
    assert_eq!(snapshot["attempted_step"], 1);
    assert_eq!(snapshot["applied_updates"], 1);
    assert_eq!(
        snapshot["global_gradient_norm"].as_f64().unwrap(),
        f64::from(stats.optimizer.grad_norm)
    );
    println!("live_gradient_cache_verified={snapshot}");
    Ok(())
}
