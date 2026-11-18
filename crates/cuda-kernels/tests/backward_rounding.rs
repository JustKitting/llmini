//! CUDA regressions for backward requantization: range, conditional mean and RNG.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::nvfp4::{
    Nvfp4DecodeModule, Nvfp4RowwiseDecodeTransposeArgs, Nvfp4RowwiseDeviceTensor,
};
use rust_kernels_cuda::nvfp4_quant::{MsEdenPairDeviceScaleQuantArgs, Nvfp4QuantModule};
use rust_kernels_cuda::quartet::QUARTET_MS_EDEN_SCALE_OVERRIDE;
use std::error::Error;
mod common;
type Result = std::result::Result<(), Box<dyn Error>>;
const D: usize = 128;
const N: usize = D * D;
const SIGN: u32 = 0x91ab_3327;
struct Output {
    bytes: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    trans_bytes: DeviceBuffer<u8>,
    trans_scales: DeviceBuffer<u8>,
    global: DeviceBuffer<f32>,
    chunks: DeviceBuffer<f32>,
}
impl Output {
    fn new(s: &CudaStream) -> std::result::Result<Self, cuda_core::DriverError> {
        let scale_len = rust_kernels_cuda::nvfp4::csf::rowwise_encoded_len(D, D);
        Ok(Self {
            bytes: DeviceBuffer::zeroed(s, N / 2)?,
            scales: DeviceBuffer::zeroed(s, scale_len)?,
            trans_bytes: DeviceBuffer::zeroed(s, N / 2)?,
            trans_scales: DeviceBuffer::zeroed(s, scale_len)?,
            global: DeviceBuffer::zeroed(s, 1)?,
            chunks: DeviceBuffer::zeroed(s, N / 32)?,
        })
    }
    fn quant(
        &mut self,
        s: &CudaStream,
        q: &Nvfp4QuantModule,
        x: &DeviceBuffer<f32>,
        seed: u32,
    ) -> std::result::Result<(), cuda_core::DriverError> {
        q.fp32_pair_to_nvfp4_quartet_backward_ms_eden_derived_device_scale_no_chunk_amax(
            MsEdenPairDeviceScaleQuantArgs {
                stream: s,
                x,
                out_fp4: &mut self.bytes,
                out_scales: &mut self.scales,
                transpose_out_fp4: &mut self.trans_bytes,
                transpose_out_scales: &mut self.trans_scales,
                out_chunk_amax: &mut self.chunks,
                out_global_scale: &mut self.global,
                row_count: D as u32,
                src_row_len: D as u32,
                dst_row_len: D as u32,
                transpose_dst_row_len: D as u32,
                scale_override: QUARTET_MS_EDEN_SCALE_OVERRIDE,
                sign_seed: SIGN,
                scale_seed: seed,
                transpose_scale_seed: seed ^ 0x85eb_ca6b,
                precomputed_chunk_count: None,
            },
        )
    }
    fn decode(
        &self,
        s: &CudaStream,
        decoder: &Nvfp4DecodeModule,
        trans: bool,
    ) -> std::result::Result<Vec<f32>, cuda_core::DriverError> {
        let (bytes, scales) = if trans {
            (&self.trans_bytes, &self.trans_scales)
        } else {
            (&self.bytes, &self.scales)
        };
        let mut decoded = DeviceBuffer::<f32>::zeroed(s, N)?;
        let global = self.global.to_host_vec(s)?[0];
        let globals = DeviceBuffer::from_host(s, &vec![global; D])?;
        decoder.decode_rowwise_transpose_f32(Nvfp4RowwiseDecodeTransposeArgs {
            stream: s,
            input: Nvfp4RowwiseDeviceTensor::new(bytes, scales, &globals),
            output: &mut decoded,
            rows: D as u32,
            cols: D as u32,
        })?;
        let t = decoded.to_host_vec(s)?;
        Ok((0..N).map(|i| t[(i % D) * D + i / D]).collect())
    }
}
fn hash(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^ (v >> 16)
}
fn sign(k: usize) -> f32 {
    if hash(SIGN ^ k as u32) & 1 == 0 {
        1.
    } else {
        -1.
    }
}
fn fixture() -> Vec<f32> {
    (0..N)
        .map(|i| {
            let bits = hash(i as u32 ^ 0xd301_a2f9);
            ((bits >> 8) as f32 * (1. / 16777216.) - 0.5) * (1. + (i / D % 7) as f32 * 0.125)
        })
        .collect()
}
fn transform(x: &[f32], transpose: bool) -> Vec<f32> {
    let mut out = vec![0.; N];
    for r in 0..D {
        for block in (0..D).step_by(32) {
            let mut a = [0f32; 32];
            for k in 0..32 {
                a[k] = x[if transpose {
                    (block + k) * D + r
                } else {
                    r * D + block + k
                }] * sign(block + k);
            }
            for stride in [1, 2, 4, 8, 16] {
                for base in (0..32).step_by(stride * 2) {
                    for k in 0..stride {
                        let lo = a[base + k];
                        let hi = a[base + k + stride];
                        a[base + k] = lo + hi;
                        a[base + k + stride] = lo - hi;
                    }
                }
            }
            for k in 0..32 {
                out[r * D + block + k] = a[k] * (1f32 / 32f32.sqrt());
            }
        }
    }
    out
}
#[test]
#[ignore = "requires generated sm_120a PTX and CUDA GPU 0"]
fn backward_rotation_preserves_concentrated_finite_input_range() -> Result {
    let (_, s, p) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(p.clone())?;
    let decode = Nvfp4DecodeModule::from_module(p)?;
    let mut output = Output::new(&s)?;
    for trans in [false, true] {
        let x = (0..N)
            .map(|i| sign(if trans { i / D } else { i % D }))
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&s, &x)?;
        output.quant(&s, &q, &input, 0x1247_9a31)?;
        let observed = output.decode(&s, &decode, trans)?;
        let ratio = (observed.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / N as f64).sqrt();
        eprintln!("CUDA backward concentrated rotation transpose={trans} norm_ratio={ratio:.9}");
        assert!(
            (ratio - 1.).abs() < 0.07,
            "Hadamard output clipped: transpose={trans} norm_ratio={ratio}"
        );
    }
    Ok(())
}
#[test]
#[ignore = "requires generated sm_120a PTX and CUDA GPU 0"]
fn backward_rounding_seed_changes_payload_and_replays_exactly() -> Result {
    let (_, s, p) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(p)?;
    let input = DeviceBuffer::from_host(&s, &fixture())?;
    let mut output = Output::new(&s)?;
    output.quant(&s, &q, &input, 0x52de_0047)?;
    let first = output.bytes.to_host_vec(&s)?;
    let first_t = output.trans_bytes.to_host_vec(&s)?;
    output.quant(&s, &q, &input, 0x52de_0047)?;
    assert_eq!(first, output.bytes.to_host_vec(&s)?);
    assert_eq!(first_t, output.trans_bytes.to_host_vec(&s)?);
    output.quant(&s, &q, &input, 0x952d_3371)?;
    assert!(
        first != output.bytes.to_host_vec(&s)?,
        "active row producer ignores the rounding seed"
    );
    assert!(
        first_t != output.trans_bytes.to_host_vec(&s)?,
        "active transpose producer ignores the rounding seed"
    );
    Ok(())
}
#[test]
#[ignore = "requires generated sm_120a PTX and CUDA GPU 0"]
fn backward_requantization_preserves_conditional_mean_in_both_orientations() -> Result {
    const SAMPLES: usize = 256;
    let (_, s, p) = common::cuda_test_context()?;
    let q = Nvfp4QuantModule::from_module(p.clone())?;
    let decode = Nvfp4DecodeModule::from_module(p)?;
    let x = fixture();
    let input = DeviceBuffer::from_host(&s, &x)?;
    let mut output = Output::new(&s)?;
    let mut means = [vec![0f64; N], vec![0f64; N]];
    for sample in 0..SAMPLES {
        output.quant(&s, &q, &input, hash(sample as u32 ^ 0xd251_1f53))?;
        for (orientation, mean) in means.iter_mut().enumerate() {
            let decoded = output.decode(&s, &decode, orientation == 1)?;
            for (sum, value) in mean.iter_mut().zip(decoded) {
                *sum += value as f64 / SAMPLES as f64;
            }
        }
    }
    for (orientation, mean) in means.iter().enumerate() {
        let expected = transform(&x, orientation == 1);
        let error = mean
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - *b as f64).powi(2))
            .sum::<f64>();
        let norm = expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        let relative = (error / norm).sqrt();
        eprintln!(
            "CUDA backward conditional-mean relative_l2={relative:.9} orientation={orientation}"
        );
        assert!(
            relative < 0.02,
            "biased backward requantization: orientation={orientation} relative_l2={relative}"
        );
    }
    Ok(())
}
