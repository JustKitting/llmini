//! Failure cases declared before exact-KDA preparation: the real B8/S2048/H32
//! affine-plane layout; original SiLU/norm/beta/decay; one-round native half
//! tapes; bounded incoming/exclusive-exit factors; packed B/U arena ownership;
//! zero/tiny vectors, beta limits, cumulative cap crossings and cancellation;
//! saved/unsaved equality, partial tails, padded NaNs, sequence/head boundaries,
//! complete writes, guards and pre-launch host shape rejection.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_prepare::{FactorArgs, KdaExactPrepareModule, PrepareArgs};
use std::error::Error;
mod common;
#[allow(dead_code)]
#[path = "causal_attention_backward_tc/f16.rs"]
mod half_reference;

const B: usize = 8;
const S: usize = 2048;
const H: usize = 32;
const D: usize = 2048;
const QKV: usize = 8320;
const CHUNKS: usize = B * H * (S / 64);
const N: usize = CHUNKS * 4096;
const GUARD: usize = 257;

fn raw(case: usize, t: usize, h: usize, plane: usize, batch: usize) -> f32 {
    match plane {
        0 | 1 => {
            let v =
                ((t * (13 + plane * 4) + h * (7 + plane * 2) + case) as i32 % 11 - 5) as f32 / 16.0;
            if case == 0 {
                0.0
            } else if case == 1 {
                v * 1e-7
            } else {
                v
            }
        }
        2 => ((t * 7 + h * 3 + case) as i32 % 19 - 9) as f32 / 16.0 + batch as f32 / 32.0,
        3 => match case {
            2 => {
                if t == 0 {
                    60000.0
                } else {
                    -4.600166
                }
            }
            3 => 1000.0,
            4 => -100.0,
            _ => ((t + h) % 5) as f32 / 4.0,
        },
        _ => unreachable!(),
    }
}
fn raw_beta(case: usize, t: usize) -> f32 {
    match case {
        6 => -100.0,
        7 => 100.0,
        _ => (t as i32 % 7 - 3) as f32 / 4.0,
    }
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn gate(x: f64) -> f64 {
    -0.01 * if x > 20.0 { x } else { x.exp().ln_1p() }
}
fn half_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 255) as i32 - 127 + 15;
    let frac = bits & 0x7fffff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp < -10 {
        return sign;
    }
    let (base, remainder, halfway) = if exp <= 0 {
        let mant = frac | 0x800000;
        let shift = (14 - exp) as u32;
        (mant >> shift, mant & ((1 << shift) - 1), 1 << (shift - 1))
    } else {
        (((exp as u32) << 10) | (frac >> 13), frac & 8191, 4096)
    };
    (sign as u32 + base + u32::from(remainder > halfway || (remainder == halfway && base & 1 == 1)))
        as u16
}
fn decode(x: u16) -> f64 {
    f64::from(half_reference::f16_bits_to_f32(x))
}
fn check_half(bits: u16, expected: f64, context: &str) {
    let rounded = decode(half_bits(expected as f32));
    let spacing = if rounded.abs() < 2.0f64.powi(-14) {
        2.0f64.powi(-24)
    } else {
        2.0f64.powi(rounded.abs().log2().floor() as i32 - 10)
    };
    assert!(
        decode(bits).is_finite() && (decode(bits) - rounded).abs() <= 1e-7 + spacing,
        "{context} actual={} expected={expected} rounded={rounded}",
        decode(bits)
    );
}
struct Buffers {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    v: DeviceBuffer<u16>,
    qg: DeviceBuffer<u16>,
    kg: DeviceBuffer<u16>,
    bu: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
}
impl Buffers {
    fn new(stream: &CudaStream, n: usize) -> Result<Self, cuda_core::DriverError> {
        let fp = || DeviceBuffer::from_host(stream, &vec![f32::from_bits(0x7fc01234); n + GUARD]);
        let half = || DeviceBuffer::from_host(stream, &vec![0x7e12u16; n + GUARD]);
        Ok(Self {
            q: fp()?,
            k: fp()?,
            g: fp()?,
            beta: DeviceBuffer::from_host(
                stream,
                &vec![f32::from_bits(0x7fc01234); n / 64 + GUARD],
            )?,
            v: half()?,
            qg: half()?,
            kg: half()?,
            bu: fp()?,
            carry: DeviceBuffer::from_host(
                stream,
                &vec![f32::from_bits(0x7fc01234); n / 64 + GUARD],
            )?,
        })
    }
    fn prepare(
        &mut self,
        module: &KdaExactPrepareModule,
        stream: &CudaStream,
        input: &DeviceBuffer<f32>,
        tape: Option<&mut DeviceBuffer<u16>>,
        b: u32,
        s: u32,
        h: u32,
        qkv: u32,
    ) -> Result<(), cuda_core::DriverError> {
        module.prepare(PrepareArgs {
            stream,
            qkv: input,
            qkv_f16: tape,
            q: &mut self.q,
            k: &mut self.k,
            g: &mut self.g,
            beta: &mut self.beta,
            vbeta: &mut self.v,
            batch_size: b,
            seq_len: s,
            head_count: h,
            qkv_dim: qkv,
        })
    }
    fn factors(
        &mut self,
        module: &KdaExactPrepareModule,
        stream: &CudaStream,
        bh: u32,
        s: u32,
    ) -> Result<(), cuda_core::DriverError> {
        module.factors(FactorArgs {
            stream,
            q: &self.q,
            k: &self.k,
            g: &self.g,
            beta: &self.beta,
            qg: &mut self.qg,
            kg: &mut self.kg,
            incoming_rhs_u: &mut self.bu,
            exit_decay: &mut self.carry,
            batch_heads: bh,
            seq_len: s,
        })
    }
}

#[test]
#[ignore = "native B8 raw/gate/bounded-factor preparation and sanitizer gate"]
fn native_kda_exact_prepare_original_planes_bounded_factors_and_tails() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(KdaExactPrepareModule::from_module)?;
    let mut input = vec![-0.375f32; B * S * QKV];
    for b in 0..B {
        for t in 0..S {
            for head in 0..H {
                let case = (b * H + head) % 8;
                for dim in 0..64 {
                    for plane in 0..4 {
                        input[(b * S + t) * QKV + plane * D + head * 64 + dim] =
                            raw(case, t % 64, dim, plane, b * H + head);
                    }
                }
                input[(b * S + t) * QKV + 4 * D + head] = raw_beta(case, t % 64);
            }
        }
    }
    let qkv = DeviceBuffer::from_host(&stream, &input)?;
    let mut tape = DeviceBuffer::from_host(&stream, &vec![0x7e12u16; input.len() + GUARD])?;
    let mut out = Buffers::new(&stream, N)?;
    out.prepare(
        &module,
        &stream,
        &qkv,
        Some(&mut tape),
        B as u32,
        S as u32,
        H as u32,
        QKV as u32,
    )?;
    out.factors(&module, &stream, (B * H) as u32, S as u32)?;
    let saved = tape.to_host_vec(&stream)?;
    for (i, (&actual, &x)) in saved[..input.len()].iter().zip(&input).enumerate() {
        assert_eq!(actual, half_bits(x), "original affine-plane tape index={i}");
    }
    assert!(saved[input.len()..].iter().all(|&x| x == 0x7e12));
    drop(saved);
    let q = out.q.to_host_vec(&stream)?;
    let k = out.k.to_host_vec(&stream)?;
    let g = out.g.to_host_vec(&stream)?;
    let beta = out.beta.to_host_vec(&stream)?;
    let v = out.v.to_host_vec(&stream)?;
    let qg = out.qg.to_host_vec(&stream)?;
    let kg = out.kg.to_host_vec(&stream)?;
    let bu = out.bu.to_host_vec(&stream)?;
    let carry = out.carry.to_host_vec(&stream)?;
    let norms = (0..8 * 64)
        .map(|i| {
            let case = i / 64;
            let t = i % 64;
            let mut qs = 0.0;
            let mut ks = 0.0;
            for h in 0..64 {
                qs += silu(f64::from(raw(case, t, h, 0, 0))).powi(2);
                ks += silu(f64::from(raw(case, t, h, 1, 0))).powi(2);
            }
            ((qs + 1e-6).sqrt() + 1e-6, (ks + 1e-6).sqrt() + 1e-6)
        })
        .collect::<Vec<_>>();
    let mut max_qk_error = 0.0f64;
    let mut max_g_error = 0.0f64;
    for chunk in 0..CHUNKS {
        let bh = chunk / (S / 64);
        let b = bh / H;
        let case = bh % 8;
        for h in 0..64 {
            let mut prefix = 0.0f32;
            let logs = (0..64)
                .map(|t| g[chunk * 4096 + t * 64 + h])
                .collect::<Vec<_>>();
            let mut suffix = vec![0.0f32; 64];
            let mut acc = 0.0;
            for t in (0..64).rev() {
                suffix[t] = acc;
                acc += logs[t];
            }
            for t in 0..64 {
                let i = chunk * 4096 + t * 64 + h;
                let (qn, kn) = norms[case * 64 + t];
                let eq = silu(f64::from(raw(case, t, h, 0, b))) * 0.125 / qn;
                let ek = silu(f64::from(raw(case, t, h, 1, b))) / kn;
                let eg = gate(f64::from(raw(case, t, h, 3, b)));
                let eb = sigmoid(f64::from(raw_beta(case, t)));
                max_qk_error = max_qk_error
                    .max((f64::from(q[i]) - eq).abs())
                    .max((f64::from(k[i]) - ek).abs());
                max_g_error = max_g_error.max((f64::from(g[i]) - eg).abs());
                assert!(
                    (f64::from(q[i]) - eq).abs() < 2e-6 && (f64::from(k[i]) - ek).abs() < 2e-6,
                    "q/k original activation/normalization"
                );
                assert!(
                    (f64::from(g[i]) - eg).abs() < 2e-6 + 2e-7 * eg.abs(),
                    "original log gate"
                );
                assert!(
                    (f64::from(beta[chunk * 64 + t]) - eb).abs() < 2e-7,
                    "original beta gate"
                );
                check_half(
                    v[i],
                    silu(f64::from(raw(case, t, h, 2, bh))) * eb,
                    "one-round Vbeta",
                );
                prefix += logs[t];
                check_half(
                    qg[i],
                    f64::from(q[i]) * f64::from(prefix).exp(),
                    "bounded incoming readout",
                );
                check_half(
                    kg[i],
                    f64::from(k[i]) * f64::from(suffix[t]).exp(),
                    "exclusive terminal key",
                );
                let packed = bu[chunk * 4096 + (t * 64 + h) / 2].to_bits();
                let bits = if h % 2 == 0 {
                    (packed & 65535) as u16
                } else {
                    (packed >> 16) as u16
                };
                check_half(
                    bits,
                    f64::from(beta[chunk * 64 + t] * k[i]) * f64::from(prefix).exp(),
                    "bounded B packed arena",
                );
            }
            let expected = f64::from(logs.iter().copied().sum::<f32>()).exp();
            let actual = f64::from(carry[chunk * 64 + h]);
            assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
            assert!((actual - expected).abs() <= 2e-7, "true terminal carry");
        }
        assert!(
            bu[chunk * 4096 + 2048..(chunk + 1) * 4096]
                .iter()
                .all(|x| x.to_bits() == 0x7fc01234),
            "B producer owns only first half of each U arena"
        );
    }
    for data in [&q, &k, &g, &bu] {
        assert!(data[N..].iter().all(|x| x.to_bits() == 0x7fc01234));
    }
    for data in [&qg, &kg, &v] {
        assert!(data[N..].iter().all(|&x| x == 0x7e12));
    }
    assert!(
        beta[N / 64..]
            .iter()
            .chain(&carry[N / 64..])
            .all(|x| x.to_bits() == 0x7fc01234)
    );
    // Save/no-save must use exactly the same original FP32 activation path.
    out.prepare(
        &module, &stream, &qkv, None, B as u32, S as u32, H as u32, QKV as u32,
    )?;
    for (device, expected) in [(&out.q, &q), (&out.k, &k), (&out.g, &g), (&out.beta, &beta)] {
        assert!(
            device
                .to_host_vec(&stream)?
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert_eq!(out.v.to_host_vec(&stream)?, v);
    // Partial sequences use unpadded input rows and padded output chunks. A
    // different second sequence catches sequence/head/chunk stride mistakes.
    for len in [1usize, 15, 16, 17, 63, 64, 65, 127] {
        let n = 2 * 2 * len.div_ceil(64) * 4096;
        let stride = 4 * 128 + 2 + 14;
        let small = (0..2 * len * stride)
            .map(|i| ((i * 13) % 23) as f32 / 32.0 - 0.25)
            .collect::<Vec<_>>();
        let small_dev = DeviceBuffer::from_host(&stream, &small)?;
        let mut a = Buffers::new(&stream, n)?;
        a.prepare(
            &module,
            &stream,
            &small_dev,
            None,
            2,
            len as u32,
            2,
            stride as u32,
        )?;
        for data in [&a.q, &a.k, &a.g] {
            let values = data.to_host_vec(&stream)?;
            assert!(values[..n].iter().all(|x| x.is_finite()));
            let chunks = len.div_ceil(64);
            for bh in 0..4 {
                for local in len..chunks * 64 {
                    assert_eq!(values[(bh * chunks * 64 + local) * 64].to_bits(), 0);
                }
            }
        }
        let ah = a.q.to_host_vec(&stream)?;
        for b in 0..2 {
            for head in 0..2 {
                for t in 0..len {
                    let row = b * len + t;
                    let mut sum = 0.0;
                    for h in 0..64 {
                        sum += silu(f64::from(small[row * stride + head * 64 + h])).powi(2);
                    }
                    let expected = silu(f64::from(small[row * stride + head * 64])) * 0.125
                        / ((sum + 1e-6).sqrt() + 1e-6);
                    let index = (((b * 2 + head) * len.div_ceil(64) * 64 + t) * 64) as usize;
                    assert!(
                        (f64::from(ah[index]) - expected).abs() < 2e-6,
                        "partial original row/sequence stride"
                    );
                }
            }
        }
        let chunks = len.div_ceil(64);
        for data in [&mut a.q, &mut a.k, &mut a.g] {
            let mut poisoned = data.to_host_vec(&stream)?;
            for bh in 0..4 {
                for local in len..chunks * 64 {
                    poisoned[(bh * chunks * 64 + local) * 64..(bh * chunks * 64 + local + 1) * 64]
                        .fill(f32::NAN);
                }
            }
            *data = DeviceBuffer::from_host(&stream, &poisoned)?;
        }
        a.factors(&module, &stream, 4, len as u32)?;
        for data in [&a.qg, &a.kg] {
            let values = data.to_host_vec(&stream)?;
            assert!(
                values[..n].iter().all(|&x| decode(x).is_finite()),
                "poisoned padding entered a retained factor"
            );
            for bh in 0..4 {
                for local in len..chunks * 64 {
                    assert!(
                        values
                            [(bh * chunks * 64 + local) * 64..(bh * chunks * 64 + local + 1) * 64]
                            .iter()
                            .all(|&x| x == 0)
                    );
                }
            }
        }
        assert!(
            a.carry.to_host_vec(&stream)?[..n / 64]
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
        );
    }
    for (b, s, h, qkv_dim) in [
        (0, 2048, 32, 8320),
        (8, 0, 32, 8320),
        (8, 2048, 0, 8320),
        (8, 2048, 32, 8192),
        (u32::MAX, 2048, 32, 8320),
        (8, u32::MAX, 32, 8320),
    ] {
        assert!(
            out.prepare(&module, &stream, &qkv, None, b, s, h, qkv_dim)
                .is_err()
        );
    }
    let short = DeviceBuffer::from_host(&stream, &[0.0f32])?;
    assert!(
        out.prepare(&module, &stream, &short, None, 8, 2048, 32, 8320)
            .is_err()
    );
    assert!(
        out.q
            .to_host_vec(&stream)?
            .iter()
            .zip(&q)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!(
        "native_exact_KDA_prepare_verified B=8 S=2048 H=32 chunk_batch={CHUNKS} raw_planes=4 beta=original partial_lengths=8 saved_unsaved=bit_exact max_qk_error={max_qk_error:e} max_log_gate_error={max_g_error:e} B_arena_unused=unchanged guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
