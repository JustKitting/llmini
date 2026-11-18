pub const QUARTET_MS_EDEN_SCALE_OVERRIDE: f32 = (17.0 / 16.0) * 0.93;
pub const QUARTET_MS_EDEN_FP8_MAX: f32 = 256.0;
pub const QUARTET_MS_EDEN_FP4_MAX: f32 = 6.0;

#[inline(always)]
pub fn quartet_backward_ms_eden_global_scale(amax: f32) -> f32 {
    if amax == 0.0 {
        1.0
    } else {
        // A normalized 32-point Hadamard can concentrate a finite input by
        // sqrt(32). Reserve that range before selecting local E4M3 scales.
        // The small factor covers the five FP32 butterfly rounding stages.
        amax * (5.656_854_249 * (1.0 + 32.0 * f32::EPSILON) / (448.0 * 6.0))
    }
}
