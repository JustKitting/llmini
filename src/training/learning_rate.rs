mod config;
mod schedule;

pub(super) use config::{adam_scale, next_latent_scale};
pub(super) use schedule::{
    adam_multiplier, muon_multiplier, next_latent_adam_multiplier,
    schedule_free_average_coefficient, schedule_free_beta, warmup_multiplier,
};
