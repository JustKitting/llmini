use std::time::{Duration, Instant};

pub(super) struct WallClockBudget {
    start: Instant,
    max: Duration,
    excluded: Duration,
}

impl WallClockBudget {
    pub(super) fn new(max_seconds: f64) -> Self {
        Self {
            start: Instant::now(),
            max: Duration::from_secs_f64(max_seconds),
            excluded: Duration::ZERO,
        }
    }

    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    pub(super) fn expired(&self) -> bool {
        self.elapsed() >= self.max
    }

    pub(super) fn exclude(&mut self, duration: Duration) {
        self.excluded = self.excluded.saturating_add(duration);
    }

    fn elapsed(&self) -> Duration {
        self.start.elapsed().saturating_sub(self.excluded)
    }
}
