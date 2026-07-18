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

const E2M1_ZERO: u8 = 0x0;
const E2M1_POS_MIN: u8 = 0x1;
const E2M1_ONE: u8 = 0x2;
const E2M1_NEG_MIN: u8 = 0x9;
const E4M3_ONE: u8 = 0x38;

const fn pack_e2m1_pair(lo: u8, hi: u8) -> u8 {
    (lo & 0x0f) | ((hi & 0x0f) << 4)
}
