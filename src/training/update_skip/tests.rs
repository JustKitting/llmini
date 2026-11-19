use super::{UpdateSkipConfig, UpdateSkipState};

fn state() -> UpdateSkipState {
    UpdateSkipState::from_config(UpdateSkipConfig::for_test())
}

#[test]
fn waits_for_minimum_history() {
    let mut state = state();
    assert!(!state.observe(Some(1.0), 1.0).skipped);
    assert!(!state.observe(Some(100.0), 100.0).skipped);
}

#[test]
fn skips_loss_outlier_after_history() {
    let mut state = state();
    assert!(!state.observe(Some(1.0), 1.0).skipped);
    assert!(!state.observe(Some(1.1), 1.0).skipped);
    let decision = state.observe(Some(10.0), 1.0);
    assert!(decision.skipped);
    assert!(decision.loss_spike);
    assert!(!decision.grad_norm_spike);
}

#[test]
fn skips_grad_norm_outlier_after_history() {
    let mut state = state();
    assert!(!state.observe(None, 1.0).skipped);
    assert!(!state.observe(None, 1.1).skipped);
    let decision = state.observe(None, 10.0);
    assert!(decision.skipped);
    assert!(!decision.loss_spike);
    assert!(decision.grad_norm_spike);
}

#[test]
fn capped_history_rejects_a_second_gradient_spike() {
    let mut baseline = state();
    let mut config = UpdateSkipConfig::for_test();
    config.cap_grad_history = true;
    let mut capped = UpdateSkipState::from_config(config);

    for state in [&mut baseline, &mut capped] {
        assert!(!state.observe(None, 1.0).skipped);
        assert!(!state.observe(None, 1.1).skipped);
        assert!(state.observe(None, 10.0).grad_norm_spike);
    }
    assert!(!baseline.observe(None, 10.0).skipped);
    assert!(capped.observe(None, 10.0).grad_norm_spike);

    let mut flat = UpdateSkipState::from_config(config);
    assert!(!flat.observe(None, 1.0).skipped);
    assert!(!flat.observe(None, 1.0).skipped);
    assert!(flat.observe(None, 10.0).grad_norm_spike);
    assert!(!flat.observe(None, 10.0).skipped);
}

#[test]
fn skips_non_finite_without_history() {
    let mut state = state();
    for (loss, norm) in [
        (Some(f32::NAN), 1.0),
        (None, f32::NAN),
        (None, f32::INFINITY),
    ] {
        let decision = state.observe(loss, norm);
        assert!(decision.skipped && decision.non_finite);
    }
}
