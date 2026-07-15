use std::collections::HashSet;

use super::{
    analysis::SweepAnalysis, candidate::Candidate, config::SweepConfig, history::Trial,
    proposal_pool, rng::SweepRng,
};

mod proposal;
mod select;
#[cfg(test)]
mod tests;
mod trial;

pub use proposal::{Proposal, ScoredCandidate};
use select::{select_candidate, unseen_random};
use trial::{best_local_center, infeasible_build_shapes, observed_loss};

pub fn propose(
    trials: &[Trial],
    seen: &HashSet<String>,
    rng: &mut SweepRng,
    config: &SweepConfig,
    analysis: &SweepAnalysis,
    baseline: Option<&Candidate>,
) -> Proposal {
    let infeasible_builds = infeasible_build_shapes(trials, config);
    if let Some(candidate) = baseline {
        let candidate = candidate.with_model_floor();
        if !seen.contains(&candidate.key()) && !infeasible_builds.contains(&candidate.build_key()) {
            return Proposal::single("baseline", candidate, analysis, config);
        }
    }

    let completed = trials
        .iter()
        .filter(|trial| observed_loss(trial).is_some())
        .count();
    if completed < config.random_trials {
        return Proposal::single(
            "random",
            unseen_random(seen, rng, &infeasible_builds),
            analysis,
            config,
        );
    }

    let center =
        best_local_center(trials, config).or_else(|| baseline.map(Candidate::with_model_floor));
    let observed = trials
        .iter()
        .map(|trial| trial.candidate.clone())
        .collect::<Vec<_>>();
    let mut ranked = proposal_pool::sample(seen, rng, config, analysis, center.as_ref(), &observed)
        .into_iter()
        .filter(|pooled| !infeasible_builds.contains(&pooled.candidate.build_key()))
        .map(|pooled| ScoredCandidate::from_pooled(pooled, analysis, config))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.score.score.total_cmp(&a.score.score));
    let candidate = select_candidate(&ranked, rng)
        .cloned()
        .unwrap_or_else(|| unseen_random(seen, rng, &infeasible_builds));
    Proposal {
        candidate,
        reason: "model",
        ranked,
    }
}
