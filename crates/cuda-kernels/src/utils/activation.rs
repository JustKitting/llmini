//! Fixed initialization-mean shift for this project's ReLU-squared MLP.
include!(concat!(env!("OUT_DIR"), "/activation_config.rs"));
#[inline(always)]
pub fn centered_relu2_from_positive(relu: f32) -> f32 {
    relu * relu - RELU2_INITIAL_MEAN
}
#[inline(always)]
pub(crate) fn unshifted_relu2(value: f32) -> f32 {
    let positive = crate::float_ptx::max_f32(value, 0.0);
    positive * positive
}
