//! LIDAS layer ancestry and Prop-1 timing for the four-layer hybrid cadence.

use crate::config::KIMI_FULL_ATTENTION_PERIOD as GROWTH;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowthPlan {
    total_attempts: usize,
    stages: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrowthTransition {
    previous_depth: usize,
    insert_at: usize,
}

impl GrowthPlan {
    pub fn new(final_depth: usize, total_attempts: usize) -> Option<Self> {
        if final_depth < GROWTH || !final_depth.is_multiple_of(GROWTH) {
            return None;
        }
        let stages = final_depth / GROWTH;
        let weights = stages.checked_mul(stages + 1)? / 2;
        // Guarantee at least one attempt at every depth.
        (total_attempts >= weights).then_some(Self {
            total_attempts,
            stages,
        })
    }

    /// Depth for the next batch, indexed by already attempted batches. Skipped
    /// updates consume a batch but do not reset or advance the optimizer clock.
    pub fn depth(&self, completed_attempts: usize) -> usize {
        let stages = (1..self.stages)
            .find(|&stage| completed_attempts < self.stage_end(stage))
            .unwrap_or(self.stages);
        stages * GROWTH
    }

    fn stage_end(&self, stage: usize) -> usize {
        let prefix = stage as u128 * (stage as u128 + 1) / 2;
        let total = self.stages as u128 * (self.stages as u128 + 1) / 2;
        (self.total_attempts as u128 * prefix / total) as usize
    }
}

impl GrowthTransition {
    pub fn new(previous_depth: usize, next_depth: usize) -> Option<Self> {
        if previous_depth < GROWTH
            || !previous_depth.is_multiple_of(GROWTH)
            || previous_depth.checked_add(GROWTH)? != next_depth
        {
            return None;
        }
        let insert_at = (previous_depth + GROWTH) / 2;
        Some(Self {
            previous_depth,
            insert_at,
        })
    }

    pub fn depth(self) -> usize {
        self.previous_depth + GROWTH
    }

    /// Yields source and destination slots in safe in-place copy order. The
    /// existing suffix must move before the new copies overwrite those slots.
    pub fn copies(self) -> impl Iterator<Item = (usize, usize)> {
        (self.insert_at..self.depth())
            .rev()
            .map(|destination| (destination - GROWTH, destination))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uses_full_attention;

    #[test]
    fn twenty_thousand_attempts_keep_the_proportional_stage_boundaries() {
        let plan = GrowthPlan::new(16, 20_000).unwrap();
        for (attempt, depth) in [
            (0, 4),
            (1_999, 4),
            (2_000, 8),
            (5_999, 8),
            (6_000, 12),
            (11_999, 12),
            (12_000, 16),
            (20_000, 16),
        ] {
            assert_eq!(plan.depth(attempt), depth);
        }
        let applications: usize = (0..20_000).map(|attempt| plan.depth(attempt)).sum();
        assert_eq!(applications, 240_000);
    }

    #[test]
    fn in_place_insertion_preserves_the_suffix_and_the_attention_kind() {
        for (old, expected) in [
            (4, vec![0, 1, 2, 3, 0, 1, 2, 3]),
            (8, vec![0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]),
            (12, vec![0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11]),
        ] {
            let mut slots: Vec<_> = (0..old).collect();
            slots.resize(old + GROWTH, usize::MAX);
            let insertion = GrowthTransition::new(old, slots.len()).unwrap();
            for (source, destination) in insertion.copies() {
                assert_eq!(
                    uses_full_attention(source),
                    uses_full_attention(destination)
                );
                slots[destination] = slots[source];
            }
            assert_eq!(slots, expected);
        }
    }
}
