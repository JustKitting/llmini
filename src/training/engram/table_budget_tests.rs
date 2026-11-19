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

#[test]
fn four_gram_budget_serializes_all_orders_without_reinterpreting_v4() {
    for (requested, minimum) in [
        (None, 328_044_672),
        (Some(500_000_000), 500_000_000),
        (Some(2_000_000_000), 2_000_000_000),
    ] {
        let ordinary = Config::multistream_with_table_budget(requested).unwrap();
        let configured = Config::multistream_with_ngram_budget(requested, true).unwrap();
        let actual = configured.iter().map(Config::table_len).sum::<usize>();
        assert!((minimum..minimum + 1_000_000).contains(&actual));
        assert_eq!(
            actual,
            ordinary.iter().map(Config::table_len).sum::<usize>()
        );
        println!("four_gram_engram_budget requested={minimum} actual={actual}");
        let mut primes = std::collections::HashSet::new();
        for (cfg, original) in configured.iter().zip(ordinary) {
            assert_eq!((original.max_ngram(), original.memory()), (3, 1024));
            assert_eq!((cfg.max_ngram(), cfg.memory(), cfg.streams()), (4, 1536, 4));
            assert_eq!(cfg.projector_indices(), original.projector_indices());
            assert_eq!(cfg.multipliers[..3], original.multipliers);
            assert_eq!(cfg.primes.len(), 24);
            assert!(cfg.primes.iter().all(|p| primes.insert(*p)));

            let descriptor = cfg.descriptor();
            assert_eq!(descriptor.len(), 49);
            assert_eq!(descriptor[0], 5.);
            assert_eq!(descriptor[8], 4.);
            assert_eq!(
                Config::parse(&descriptor, false).unwrap().descriptor(),
                descriptor
            );
            assert!(Config::parse(&descriptor, true).is_err());
            assert!(Config::parse(&descriptor[..37], false).is_err());
            let mut old_tag = descriptor.clone();
            old_tag[0] = 4.;
            assert!(Config::parse(&old_tag, false).is_err());
            let mut legacy_dilation = descriptor.clone();
            legacy_dilation[8] = 3.;
            assert_eq!(
                Config::parse(&legacy_dilation, false).unwrap().descriptor(),
                legacy_dilation
            );
            let mut invalid_dilation = descriptor.clone();
            invalid_dilation[8] = 2.;
            assert!(Config::parse(&invalid_dilation, false).is_err());
            let mut duplicate_head = descriptor.clone();
            duplicate_head[32] = duplicate_head[9];
            assert!(Config::parse(&duplicate_head, false).is_err());
        }
    }
}

#[test]
fn three_gram_descriptors_require_the_original_dilation() {
    for cfg in [Config::reference(0), Config::reference_multistream(0)] {
        let mut descriptor = cfg.descriptor();
        assert_eq!(descriptor[8], 3.);
        descriptor[8] = 4.;
        assert!(Config::parse(&descriptor, false).is_err());
    }
}
