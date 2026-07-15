use std::path::PathBuf;

use super::super::{candidate::Candidate, history::Trial};

pub(super) fn parse_trial(line: &str) -> Option<Trial> {
    let p = line.split('\t').collect::<Vec<_>>();
    if !(16..=22).contains(&p.len()) {
        return None;
    }
    let has_nextlat_lr = matches!(p.len(), 17 | 22);
    let log_index = if has_nextlat_lr { 16 } else { 15 };
    Some(Trial {
        status: p[0].to_string(),
        val_loss: parse_loss(p[1]),
        completed_steps: p[2].parse().ok(),
        candidate: parse_candidate(&p, has_nextlat_lr)?,
        log_path: PathBuf::from(p[log_index]),
        elapsed_s: p.get(log_index + 1).and_then(|value| parse_loss(value)),
        screen_val_loss: p.get(log_index + 2).and_then(|value| parse_loss(value)),
        screen_completed_steps: p.get(log_index + 3).and_then(|value| value.parse().ok()),
        screen_elapsed_s: p.get(log_index + 4).and_then(|value| parse_loss(value)),
        screen_reason: p.get(log_index + 5).and_then(|value| non_empty(value)),
    })
}

fn parse_candidate(p: &[&str], has_nextlat_lr: bool) -> Option<Candidate> {
    let warmup_index = if has_nextlat_lr { 12 } else { 11 };
    Some(Candidate {
        batch_size: p[3].parse().ok()?,
        n_layer: p[4].parse().ok()?,
        n_embd: p[5].parse().ok()?,
        n_head: p[6].parse().ok()?,
        muon_phases: p[7].parse().ok()?,
        muon_blocks: p[8].parse().ok()?,
        lr_scale: p[9].parse().ok()?,
        adam_lr_scale: p[10].parse().ok()?,
        nextlat_lr_scale: if has_nextlat_lr {
            p[11].parse().ok()?
        } else {
            1.0
        },
        warmup_steps: p[warmup_index].parse().ok()?,
        start_ratio: p[warmup_index + 1].parse().ok()?,
        amuse_beta1: p[warmup_index + 2].parse().ok()?,
        amuse_rho: p[warmup_index + 3].parse().ok()?,
    })
}

fn parse_loss(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn non_empty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}
