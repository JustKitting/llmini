use std::ops::Range;

use cuda_core::DeviceBuffer;

use super::{BOUNDARIES, WINDOW, state::Checkpoint};

pub(in crate::training) struct ReplayWindows {
    starts: Vec<usize>,
    by_boundary: [usize; BOUNDARIES + 1],
}

impl ReplayWindows {
    pub(super) fn new(engram_boundaries: impl IntoIterator<Item = usize>) -> Self {
        let mut starts = vec![0];
        starts.extend((2..=BOUNDARIES).step_by(WINDOW));
        // Replaying mHC post maps alone cannot reproduce Engram injections.
        // Save the already-injected stream where an Engram module runs.
        starts.extend(engram_boundaries);
        starts.sort_unstable();
        starts.dedup();
        let by_boundary = std::array::from_fn(|index| starts.partition_point(|&s| s <= index) - 1);
        Self {
            starts,
            by_boundary,
        }
    }

    pub(super) fn starts(&self) -> &[usize] {
        &self.starts
    }

    pub(super) fn checkpoint_slot(&self, index: usize) -> usize {
        self.by_boundary[index]
    }

    pub(super) fn range(&self, index: usize) -> Range<usize> {
        let checkpoint = self.checkpoint_slot(index);
        self.starts[checkpoint]
            ..self
                .starts
                .get(checkpoint + 1)
                .copied()
                .unwrap_or(BOUNDARIES + 1)
    }

    pub(super) fn tape_slot(&self, index: usize) -> usize {
        index - self.starts[self.checkpoint_slot(index)]
    }

    pub(in crate::training) fn input<'a>(
        &self,
        checkpoints: &'a [Checkpoint],
        replay_inputs: &'a [DeviceBuffer<f32>],
        index: usize,
    ) -> &'a DeviceBuffer<f32> {
        let slot = self.tape_slot(index);
        if slot == 0 {
            &checkpoints[self.checkpoint_slot(index)].input
        } else {
            &replay_inputs[slot - 1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engram_insertions_start_replay_windows_without_moving_other_checkpoints() {
        assert!(BOUNDARIES >= 18);
        let original = ReplayWindows::new([]);
        let legacy = ReplayWindows::new([2, 10]);
        let aligned = ReplayWindows::new([2, 14]);
        assert_eq!(original.starts(), legacy.starts());
        assert_eq!(original.starts(), aligned.starts());

        for insertion in (4..BOUNDARIES).step_by(2) {
            let shifted = ReplayWindows::new([2, insertion]);
            assert!(
                original
                    .starts()
                    .iter()
                    .all(|start| shifted.starts().contains(start))
            );
            assert_eq!(shifted.range(insertion).start, insertion);
            assert_eq!(shifted.tape_slot(insertion), 0);
            for index in 0..=BOUNDARIES {
                let range = shifted.range(index);
                assert!(range.contains(&index));
                assert!(range.len() <= WINDOW);
                assert_eq!(shifted.tape_slot(index), index - range.start);
                assert_eq!(
                    shifted.starts()[shifted.checkpoint_slot(index)],
                    range.start
                );
            }
        }
    }
}
