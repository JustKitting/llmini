mod adaptive;
mod args;
mod kernels;
mod launcher;
pub mod config {
    include!(concat!(env!("OUT_DIR"), "/mlp_config.rs"));
}

pub use args::{
    MlpBlockTopKRouteArgs, MlpDownResidualArgs, MlpUpRelu2Args, Relu2BackwardArgs,
    Relu2BackwardF16Args,
};
pub use launcher::{MlpModule, relu2_backward_amax_chunks};
