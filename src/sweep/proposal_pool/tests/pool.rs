use crate::sweep::test_fixtures::{basic_candidate as candidate, success_trial as trial};

use super::{changed_factors, config, sample_pool, wide_candidate};

#[test]
fn guided_pool_uses_main_effect_direction() {
    let config = config();
    let trials = [
        trial(candidate(4, 4), 5.0),
        trial(candidate(4, 8), 3.0),
        trial(candidate(16, 4), 3.0),
        trial(candidate(16, 8), 1.0),
    ];
    let center = candidate(8, 4);
    let pool = sample_pool(0x1234, &config, &trials, Some(&center));

    assert_eq!(pool[0].source, "guided");
    assert!(pool[0].candidate.batch_size > center.batch_size);
    assert!(pool[0].candidate.batch_size < 32);
    assert_eq!(pool[0].candidate.n_layer, 16);
    assert!(pool.iter().any(|candidate| candidate.source == "factorial"));
    assert!(pool.iter().any(|candidate| candidate.source == "local"));
    assert!(pool.iter().any(|candidate| candidate.source == "variance"));
    assert!(pool.iter().any(|candidate| candidate.source == "coverage"));
    assert!(pool.iter().any(|candidate| candidate.source == "random"));
}

#[test]
fn local_pool_refines_near_center_hyperparameters() {
    let config = config();
    let trials = (0..32)
        .map(|i| trial(wide_candidate(i), 32.0 - i as f64))
        .collect::<Vec<_>>();
    let mut center = candidate(16, 4);
    center.lr_scale = 2.309_529;
    center.adam_lr_scale = 1.626_648;
    center.nextlat_lr_scale = 1.245_083;
    center.warmup_steps = 87;
    center.start_ratio = 0.183_570;
    center.amuse_beta1 = 0.443_495;
    center.amuse_rho = 0.768_398;
    let pool = sample_pool(0x9933, &config, &trials, Some(&center));
    let locals = pool
        .iter()
        .filter(|candidate| candidate.source == "local")
        .collect::<Vec<_>>();

    assert!(!locals.is_empty());
    assert!(locals.iter().any(|local| {
        local.candidate.build_key() == center.build_key()
            && local.candidate.lr_scale != center.lr_scale
            && local.candidate.adam_lr_scale != center.adam_lr_scale
    }));
    assert!(locals.iter().all(|local| local.candidate.batch_size <= 20));
}

#[test]
fn factorial_pool_can_probe_more_than_four_supported_factors() {
    let config = config();
    let trials = (0..24)
        .map(|i| trial(wide_candidate(i), 24.0 - i as f64))
        .collect::<Vec<_>>();
    let center = wide_candidate(0);
    let pool = sample_pool(0x8822, &config, &trials, Some(&center));
    let factorial = pool
        .iter()
        .find(|candidate| candidate.source == "factorial")
        .unwrap();

    assert!(changed_factors(&center, &factorial.candidate) > 4);
}
