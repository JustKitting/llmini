#[inline(always)]
pub(super) fn random_sign_mask(seed: u32, index: u32) -> u32 {
    (hash_u32(seed ^ index) & 1) << 31
}

#[inline(always)]
pub(super) fn apply_sign_mask(value: f32, sign_mask: u32) -> f32 {
    f32::from_bits(value.to_bits() ^ sign_mask)
}

#[inline(always)]
pub(super) fn apply_random_sign(value: f32, seed: u32, index: u32) -> f32 {
    apply_sign_mask(value, random_sign_mask(seed, index))
}

#[inline(always)]
pub(super) fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}
