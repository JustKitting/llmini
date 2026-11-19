#[macro_use]
mod metric_fields;

mod activation_inspection;
mod feedback_depth_inspection;
#[cfg(test)]
mod value_rms_tests;
#[cfg(test)]
mod full_kda_tests;
#[cfg(test)]
mod stationary_rope_tests;
#[cfg(test)]
mod update_rejection_tests;
#[cfg(test)]
mod kda_decay_inspection_tests;
mod backward;
mod batch;
mod buffers;
#[cfg(test)]
mod bounded_swiglu_tests;
mod data;
mod debug_metrics;
mod device_buffer;
mod diagnostics;
pub(crate) mod engram;
mod env;
mod eval;
mod forward;
mod forward_tma;
mod generate;
mod grad_block;
mod grad_clip;
mod hnet;
mod gradient_inspection;
mod grads;
pub(crate) mod latent_feedback;
mod launch;
mod learning_rate;
mod metric_accumulator;
pub(crate) mod mhc;
mod next_latent;
mod numeric_metric;
mod optimizer;
mod optimizer_apply;
mod optimizer_muon;
mod optimizer_state;
mod optimizer_tc_scratch;
mod reference_eval;
pub(crate) mod runtime;
mod save;
mod schedule_free;
mod scratch;
mod stats;
mod symexp_lin;
mod tape;
mod tape_block;
mod tape_leaf;
mod text_metric;
mod trainer;
mod update_skip;
#[cfg(test)]
mod weight_cache_tests;

pub use batch::{ReusableTokenBatch, TokenBatch};
pub use data::TokenDataLoader;
pub use generate::SamplingConfig;
pub(crate) use launch::launch_from_args;
pub use stats::{OptimizerTrace, TrainStats};
pub use trainer::Trainer;
