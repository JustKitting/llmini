//! Fixed initialization-mean shift for this project's ReLU-squared MLP.
include!(concat!(env!("OUT_DIR"),"/activation_config.rs"));
#[inline(always)]
pub fn centered_relu2_from_positive(relu:f32)->f32 {relu*relu-RELU2_INITIAL_MEAN}
