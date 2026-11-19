//! Independent host-only IEEE binary16 oracle, never used by training.
pub fn round(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return value;
    }
    let absolute = value.abs();
    let rounded = if absolute >= 65520. {
        f32::INFINITY
    } else if absolute < 0.000_061_035_156_25 {
        (absolute * 16_777_216.).round_ties_even() / 16_777_216.
    } else {
        f32::from_bits((magnitude + 0x0fff + ((magnitude >> 13) & 1)) & !0x1fff)
    };
    rounded.copysign(value)
}

pub fn decode(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    if exponent == 0 {
        (f32::from(mantissa) / 16_777_216.).copysign(f32::from_bits(sign))
    } else {
        let exponent = if exponent == 31 {
            255
        } else {
            u32::from(exponent) + 112
        };
        f32::from_bits(sign | (exponent << 23) | (u32::from(mantissa) << 13))
    }
}

#[test]
fn all_half_encodings_are_fixed_points() {
    for bits in 0..=u16::MAX {
        let value = decode(bits);
        if value.is_nan() {
            assert!(round(value).is_nan());
        } else {
            assert_eq!(round(value).to_bits(), value.to_bits(), "half={bits:04x}");
        }
    }
}

#[test]
fn every_adjacent_finite_half_midpoint_uses_ties_to_even() {
    for bits in 0..0x7bff {
        let low = decode(bits);
        let high = decode(bits + 1);
        let midpoint = (low + high) * 0.5;
        let expected = if bits & 1 == 0 { low } else { high };
        assert_eq!(round(midpoint), expected, "half={bits:04x}");
        assert_eq!(round(-midpoint).to_bits(), (-expected).to_bits());
        assert_eq!(round(f32::from_bits(midpoint.to_bits() - 1)), low);
        assert_eq!(round(f32::from_bits(midpoint.to_bits() + 1)), high);
    }
    assert_eq!(round(65520.), f32::INFINITY);
    assert_eq!(round(-65520.), f32::NEG_INFINITY);
    assert_eq!(round(f32::from_bits(65520f32.to_bits() - 1)), 65504.);
}
