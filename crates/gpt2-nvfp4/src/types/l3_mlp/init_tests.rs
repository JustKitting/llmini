//! Host verification of actual packed MLP constructor outputs; no CUDA/training.
use super::MlpWeights;
use crate::{Gpt2Rng, Nvfp4Shape, Nvfp4Tensor};

fn fingerprint<S: Nvfp4Shape>(t: &Nvfp4Tensor<S>, hash: &mut u64) {
    for byte in t.bytes.as_ref().iter().chain(t.scales.as_ref()) {
        *hash = (*hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    for byte in t.global_scale.to_bits().to_le_bytes() {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
}
fn check_moments(bytes: &[u8], global_scale: f32, label: &str) {
    const MAGNITUDES: [f64; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
    let mut sum = 0.;
    let mut square = 0.;
    let mut count = 0usize;
    let mut outer_codes = 0usize;
    let mut bins = [0usize; 8];
    for packed in bytes {
        for code in [packed & 15, packed >> 4] {
            let magnitude = MAGNITUDES[usize::from(code & 7)];
            let value = magnitude * f64::from(global_scale) * if code & 8 == 0 { 1. } else { -1. };
            sum += value;
            square += value * value;
            count += 1;
            outer_codes += usize::from(magnitude >= 1.);
            bins[usize::from(code & 7)] += 1;
        }
    }
    let mean = sum / count as f64;
    let rms = (square / count as f64).sqrt();
    assert!(mean.abs() < 0.0001, "{label}: nonzero mean={mean}");
    assert!(
        (0.019..=0.021).contains(&rms),
        "{label}: source-normal RMS={rms}; expected native quantization of std0.02"
    );
    assert!(
        outer_codes * 10 > count,
        "{label}: initializer remained restricted to legacy three values"
    );
    // Independent normal-CDF interval probabilities at the E2M1 midpoints.
    // These are probabilities, not the integer sampling thresholds.
    let expected = [
        0.19741265136584743,
        0.34933264388041618,
        0.24195515742002582,
        0.13118123360607636,
        0.06769898307608191,
        0.011954072493481238,
        0.00046468485492734413,
        0.0000005733031437360481,
    ];
    for (index, probability) in expected.into_iter().enumerate() {
        let observed = bins[index] as f64 / count as f64;
        let tolerance = 8. * (probability * (1. - probability) / count as f64).sqrt() + 0.000002;
        assert!(
            (observed - probability).abs() <= tolerance,
            "{label}: Gaussian magnitude bin{index}: observed={observed} expected={probability} tolerance={tolerance}"
        );
    }
    println!(
        "packed_source_normal_moments label={label} values={count} mean={mean:.9} rms={rms:.9} outer_codes={outer_codes}"
    );
}

#[test]
#[ignore = "actual full-width source-normal packed constructor verification; no CUDA or CPU training"]
fn actual_bounded_mlp_constructor_source_normal_nvfp4_distribution() {
    assert!(crate::GPT2_BOUNDED_SWIGLU);
    assert_eq!((crate::GPT2_MLP_UP, crate::GPT2_MLP), (11008, 5504));
    let mut rng = Gpt2Rng::new(0x4750_5432);
    let weights = MlpWeights::init(&mut rng, 0.02);
    let paired = weights.c_fc.weight.bytes.as_ref();
    let half = paired.len() / 2;
    check_moments(&paired[..half], weights.c_fc.weight.global_scale, "gate");
    check_moments(&paired[half..], weights.c_fc.weight.global_scale, "up");
    check_moments(
        weights.c_proj.weight.bytes.as_ref(),
        weights.c_proj.weight.global_scale,
        "down",
    );
    assert_eq!(weights.c_fc.weight.global_scale, 0.02);
    assert_eq!(weights.c_proj.weight.global_scale, 0.02);
    for scales in [
        weights.c_fc.weight.scales.as_ref(),
        weights.c_proj.weight.scales.as_ref(),
    ] {
        assert!(scales.iter().all(|byte| *byte == 0x38));
    }
    assert!(
        weights
            .c_fc
            .bias
            .bytes
            .as_ref()
            .iter()
            .all(|byte| *byte == 0)
    );
    assert!(
        weights
            .c_proj
            .bias
            .bytes
            .as_ref()
            .iter()
            .all(|byte| *byte == 0)
    );
}

#[test]
#[ignore = "record actual packed constructor fingerprint and RNG advancement; no CUDA/training"]
fn actual_bounded_mlp_constructor_packed_fingerprint_and_rng() {
    assert!(crate::GPT2_BOUNDED_SWIGLU);
    let mut rng = Gpt2Rng::new(0x4750_5432);
    let weights = MlpWeights::init(&mut rng, 0.02);
    let mut payload = 0xcbf2_9ce4_8422_2325;
    fingerprint(&weights.c_fc.weight, &mut payload);
    fingerprint(&weights.c_fc.bias, &mut payload);
    fingerprint(&weights.c_proj.weight, &mut payload);
    fingerprint(&weights.c_proj.bias, &mut payload);
    println!(
        "actual_mlp_initializer_fingerprint payload={payload:016x} next_rng={:08x} paired={} product={} CPU_training=false CUDA_work=none",
        rng.next_u32(),
        crate::GPT2_MLP_UP,
        crate::GPT2_MLP
    );
}
