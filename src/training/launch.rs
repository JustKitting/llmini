#[cfg(test)]
mod bounded_engram_tests;
mod burn_shim;
mod cli;
mod config;
mod data_loader;
#[cfg(test)]
mod decay_inspection_tests;
#[cfg(test)]
mod full_credit_tests;
mod launcher;
mod loss_plot;
mod metrics;
mod output;
mod render;
#[cfg(test)]
mod stationary_engram_tests;
#[cfg(test)]
mod stationary_rope_tests;
mod strategy;
#[cfg(test)]
mod table_budget_tests;
#[cfg(test)]
mod value_engram_tests;

pub(super) use burn_shim::CudaLearningComponents;
use burn_shim::{BurnBackend, BurnInnerBackend, CudaTrainInput, CudaValidInput};
pub(super) use config::{TrainConfig, env_bool, env_nonempty};
pub(crate) use launcher::launch_from_args;
pub(super) use metrics::CudaTrainOutput;
