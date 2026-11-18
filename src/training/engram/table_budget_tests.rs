//! Metadata failure cases declared before implementing explicit capacity.
use super::{Config, LAYERS};
#[test]
fn omitted_table_budget_preserves_every_default_descriptor() {
    let configured = Config::multistream_with_table_budget(None).unwrap();
    for (i, cfg) in configured.iter().enumerate() {
        assert_eq!(
            cfg.descriptor(),
            Config::reference_multistream(i).descriptor()
        );
    }
}
#[test]
fn explicit_table_budget_preserves_full_graph_and_checkpoint_identity() {
    let configured = Config::multistream_with_table_budget(Some(500_000_000)).unwrap();
    let actual = configured.iter().map(Config::table_len).sum::<usize>();
    assert!((500_000_000..501_000_000).contains(&actual));
    let mut seen = std::collections::HashSet::new();
    for (i, cfg) in configured.iter().enumerate() {
        let old = Config::reference_multistream(i);
        assert_eq!(
            (cfg.layer, cfg.streams(), cfg.head_dim, cfg.memory()),
            (LAYERS[i], 4, 64, 1024)
        );
        assert_eq!(cfg.multipliers, old.multipliers);
        assert_eq!(cfg.gate_rms_epsilon(), old.gate_rms_epsilon());
        assert!(cfg.primes.iter().all(|p| seen.insert(*p)));
        assert_eq!(
            Config::parse(&cfg.descriptor(), false)
                .unwrap()
                .descriptor(),
            cfg.descriptor()
        );
    }
}
#[test]
fn invalid_table_budgets_are_rejected_without_cuda() {
    let default = (0..LAYERS.len())
        .map(|i| Config::reference_multistream(i).table_len())
        .sum::<usize>();
    for invalid in [0, 1, default - 1, 2_000_000_001, usize::MAX] {
        assert!(
            Config::multistream_with_table_budget(Some(invalid)).is_err(),
            "budget={invalid}"
        );
    }
}
