use crate::random::InitRng;
use crate::{Nvfp4Shape, Nvfp4Tensor};

pub(crate) trait Nvfp4ShapeInit: Nvfp4Shape + Sized {
    fn smooth_tensor(rng: &mut InitRng) -> Nvfp4Tensor<Self> {
        Self::smooth_tensor_with_global_scale(rng, 0.02)
    }

    fn smooth_tensor_with_global_scale(rng: &mut InitRng, global_scale: f32) -> Nvfp4Tensor<Self> {
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        fill_smooth_payload(bytes.as_mut(), rng);
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, global_scale)
    }

    fn reference_normal_tensor(rng: &mut InitRng, global_scale: f32) -> Nvfp4Tensor<Self> {
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        for byte in bytes.as_mut() {
            *byte = pack_e2m1_pair(normal_e2m1(rng), normal_e2m1(rng));
        }
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, global_scale)
    }

    fn zero_tensor() -> Nvfp4Tensor<Self> {
        let bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, 1.0)
    }

    fn one_tensor() -> Nvfp4Tensor<Self> {
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        bytes.as_mut().fill(pack_e2m1_pair(E2M1_ONE, E2M1_ONE));
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, 1.0)
    }

    fn scalar_tensor(value: f32) -> Nvfp4Tensor<Self> {
        assert!(value.is_finite());
        let mut bytes = Self::zero_bytes();
        let mut scales = Self::zero_scales();
        bytes.as_mut()[0] = pack_e2m1_pair(E2M1_ONE, E2M1_ZERO);
        fill_neutral_scales(scales.as_mut());
        Nvfp4Tensor::new(bytes, scales, value)
    }
}

impl<S: Nvfp4Shape> Nvfp4ShapeInit for S {}

fn fill_smooth_payload(bytes: &mut [u8], rng: &mut InitRng) {
    for byte in bytes {
        *byte = pack_e2m1_pair(smooth_e2m1(rng), smooth_e2m1(rng));
    }
}

fn fill_neutral_scales(scales: &mut [u8]) {
    scales.fill(E4M3_ONE);
}

fn smooth_e2m1(rng: &mut InitRng) -> u8 {
    let byte = rng.next_u8();
    if byte & 0x3 == 0 {
        E2M1_ZERO
    } else if byte & 0x4 == 0 {
        E2M1_POS_MIN
    } else {
        E2M1_NEG_MIN
    }
}

fn normal_e2m1(rng: &mut InitRng) -> u8 {
    // P(|Z| < midpoint) for Z~N(0,1), multiplied by2^31 and floored.
    // E2M1 magnitudes are0,.5,1,1.5,2,3,4,6; midpoint ties have zero
    // continuous probability. Only packed FP4 codes are materialized.
    const CDF: [u32; 7] = [
        423_940_440,
        1_174_126_581,
        1_693_721_325,
        1_975_430_879,
        2_120_813_338,
        2_146_484_513,
        2_147_482_416,
    ];
    // next_u8 and next_u32 each advance InitRng once. Keeping one draw per
    // weight preserves the subsequent non-MLP initialization RNG sequence.
    let draw = rng.next_u32();
    let ticket = draw & 0x7fff_ffff;
    let mut magnitude = 7u8;
    for (index, upper) in CDF.into_iter().enumerate() {
        if ticket < upper {
            magnitude = index as u8;
            break;
        }
    }
    if magnitude == 0 {
        0
    } else {
        magnitude | ((draw >> 28) as u8 & 8)
    }
}

const E2M1_ZERO: u8 = 0x0;
const E2M1_POS_MIN: u8 = 0x1;
const E2M1_ONE: u8 = 0x2;
const E2M1_NEG_MIN: u8 = 0x9;
const E4M3_ONE: u8 = 0x38;

const fn pack_e2m1_pair(lo: u8, hi: u8) -> u8 {
    (lo & 0x0f) | ((hi & 0x0f) << 4)
}
