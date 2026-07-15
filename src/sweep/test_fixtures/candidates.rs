use crate::sweep::candidate::Candidate;

pub(in crate::sweep) fn candidate(batch_size: usize, n_layer: usize, lr_scale: f64) -> Candidate {
    Candidate {
        n_embd: 2048,
        n_head: 32,
        aurora_phases: 16,
        aurora_blocks: 180,
        lr_scale,
        warmup_steps: 5,
        start_ratio: 0.0,
        ..basic_candidate(batch_size, n_layer)
    }
}

pub(in crate::sweep) fn basic_candidate(batch_size: usize, n_layer: usize) -> Candidate {
    Candidate {
        batch_size,
        n_layer,
        n_embd: 1024,
        n_head: 16,
        aurora_phases: 4,
        aurora_blocks: 80,
        lr_scale: 1.0,
        adam_lr_scale: 1.0,
        nextlat_lr_scale: 1.0,
        warmup_steps: 20,
        start_ratio: 0.1,
        amuse_beta1: 0.4,
        amuse_rho: 0.8,
    }
}

pub(in crate::sweep) fn measured_candidate() -> Candidate {
    Candidate {
        aurora_phases: 2,
        lr_scale: 1.014_040,
        adam_lr_scale: 1.980_467,
        warmup_steps: 5,
        start_ratio: 0.05,
        amuse_beta1: 0.2,
        amuse_rho: 0.5,
        ..basic_candidate(8, 2)
    }
}
