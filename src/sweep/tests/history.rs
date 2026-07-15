use crate::sweep::{baseline::Baseline, chain, history::History, trial_row};

use super::fixtures::{candidate, measured_candidate, temp_path, trial};

#[test]
fn promotes_baseline_file_when_validation_improves() {
    let path = temp_path("sweep-baseline.env");
    let mut baseline = Baseline::load(path.clone()).unwrap();

    assert!(
        baseline
            .promote_trial(&trial("success", Some(5.0), candidate(8, 16, 1.0)), false)
            .unwrap()
    );
    assert!(
        !baseline
            .promote_trial(&trial("success", Some(4.2), measured_candidate()), false)
            .unwrap()
    );
    assert!(
        baseline
            .promote_trial(&trial("success", Some(4.2), candidate(8, 16, 2.0)), false)
            .unwrap()
    );

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("VAL_LOSS=4.200000"));
    assert!(text.contains("SCREEN_LOSS=5.200000"));
    assert!(text.contains("SCREEN_COMPLETED_STEPS=10"));
    assert!(text.contains("SCREEN_ELAPSED_S=30.000000"));
    assert!(text.contains("SCREEN_REASON=screen_loss_improved"));
    assert!(text.contains("GPT2_BATCH_SIZE=8"));
    assert!(text.contains("GPT2_N_LAYER=16"));
    assert!(text.contains("GPT2_N_EMBD=2048"));
    assert!(text.contains("AURORA_MATRIX_PHASES=16"));
    assert!(text.contains("TRAIN_LR_SCALE=2.000000"));
    let loaded = Baseline::load(path.clone())
        .unwrap()
        .measured_trial()
        .unwrap();
    assert_eq!(loaded.screen_val_loss, Some(5.2));
    assert_eq!(loaded.screen_completed_steps, Some(10));
    assert_eq!(loaded.screen_elapsed_s, Some(30.0));
    assert_eq!(
        loaded.screen_reason.as_deref(),
        Some("screen_loss_improved")
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn syncs_local_real_trials_to_shared_history_once() {
    let path = temp_path("sweep-shared-history.tsv");
    let mut shared = History::load(path.clone()).unwrap();
    let trial = trial("success", Some(4.2), candidate(8, 4, 1.0));

    chain::sync_shared_history(&mut shared, std::slice::from_ref(&trial), false).unwrap();
    chain::sync_shared_history(&mut shared, &[trial], false).unwrap();

    let persisted = trial_row::read_trials(&path);
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].val_loss, Some(4.2));
    let _ = std::fs::remove_file(path);
}

#[test]
fn chains_shared_and_local_trials_without_duplicates() {
    let shared = trial("success", Some(5.0), candidate(8, 4, 1.0));
    let local_new = trial("nan", None, candidate(8, 2, 2.0));
    let local_duplicate = shared.clone();

    let trials = chain::all_trials(&[shared], &[local_duplicate, local_new]);

    assert_eq!(trials.len(), 2);
    assert_eq!(trials[0].val_loss, Some(5.0));
    assert_eq!(trials[1].status, "nan");
}
