//! Predeclared raw-gradient contract: original FP32 affine operands; exact
//! SiLU/normalizer/gate derivatives, beta outside V, tiny cotangents, separate
//! value accumulation, untouched output-gate/padding fields, actual final amax,
//! complete B8/S2048/H32, partial sequences and two-step independent FD.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_raw_vjp::{KdaExactRawVjpModule, RawVjpArgs};
use std::error::Error;
mod common;
const GUARD: usize = 257;
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
fn silu_grad(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
fn half(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    let unit = 2.0f64.powi((x.abs().log2().floor() as i32).max(-14) - 10);
    (x / unit).round_ties_even() * unit
}
#[derive(Clone)]
struct Fixture {
    b: usize,
    s: usize,
    h: usize,
    stride: usize,
    forced: Option<usize>,
    qkv: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dg: Vec<f32>,
    dv: Vec<f32>,
    db: Vec<f32>,
    initial: Vec<f32>,
}
impl Fixture {
    fn new(b: usize, s: usize, h: usize, stride: usize, forced: Option<usize>) -> Self {
        let d = h * 64;
        let chunks = s.div_ceil(64);
        let n = b * h * chunks * 4096;
        let mut f = Self {
            b,
            s,
            h,
            stride,
            forced,
            qkv: vec![-0.375; b * s * stride],
            dq: vec![f32::NAN; n],
            dk: vec![f32::NAN; n],
            dg: vec![f32::NAN; n],
            dv: vec![f32::NAN; n],
            db: vec![f32::NAN; n / 64],
            initial: vec![0.375; b * s * stride],
        };
        for batch in 0..b {
            for head in 0..h {
                let bh = batch * h + head;
                let case = forced.unwrap_or(bh % 8);
                let scale = 2.0f32.powi(if case % 2 == 0 { -28 } else { -8 });
                for t in 0..s {
                    let row = batch * s + t;
                    let compact = ((bh * chunks + t / 64) * 64 + t % 64) * 64;
                    for dim in 0..64 {
                        let q = match case {
                            0 => 0.0,
                            1 => (((t + dim) % 7) as i32 - 3) as f32 / 4096.0,
                            2 => 1.0 / 1024.0,
                            4 => 0.10003 + dim as f32 * 0.000003,
                            _ => ((t * 5 + dim * 3 + bh) % 9) as f32 / 16.0 - 0.25,
                        };
                        let k = match case {
                            0 => 0.0,
                            1 => (((t * 3 + dim) % 5) as i32 - 2) as f32 / 8192.0,
                            2 => -1.0 / 4096.0,
                            4 => -0.20007 + dim as f32 * 0.000001,
                            _ => ((t * 7 + dim * 5 + bh) % 11) as f32 / 16.0 - 0.3125,
                        };
                        let v = if case == 5 {
                            2.0 + (dim % 5) as f32 / 256.0
                        } else {
                            ((t * 3 + dim * 7 + bh) % 13) as f32 / 16.0 - 0.375
                        };
                        let g = match case {
                            0 => -100.0,
                            2 => {
                                if t % 64 == 0 {
                                    60000.0
                                } else {
                                    -4.600166
                                }
                            }
                            3 => 1000.0,
                            _ => ((t + dim) % 7) as f32 / 4.0 - 0.5,
                        };
                        for (plane, value) in [q, k, v, g].into_iter().enumerate() {
                            f.qkv[row * stride + plane * d + head * 64 + dim] = value;
                        }
                        let at = compact + dim;
                        f.dq[at] = (((t * 13 + dim * 3 + case) % 9) as i32 - 4) as f32 * scale;
                        f.dk[at] = (((t * 7 + dim * 11 + case) % 11) as i32 - 5) as f32 * scale;
                        f.dg[at] = (((t * 3 + dim * 7 + case) % 7) as i32 - 3) as f32 * scale;
                        f.dv[at] = if case == 5 {
                            0.5
                        } else {
                            (((t * 11 + dim * 5 + case) % 9) as i32 - 4) as f32 * scale
                        };
                        f.initial[row * stride + 2 * d + head * 64 + dim] =
                            ((t * 3 + dim) % 7) as f32 / 64.0 - 3.0 / 64.0;
                    }
                    f.qkv[row * stride + 4 * d + head] = match case {
                        5 => 8.0039,
                        6 => -100.0,
                        7 => 100.0,
                        _ => (t as i32 % 7 - 3) as f32 / 4.0,
                    };
                    f.db[compact / 64] = ((t + case) % 5) as f32 * scale;
                }
            }
        }
        f
    }
    fn compact(&self, row: usize, head: usize) -> usize {
        let batch = row / self.s;
        let t = row % self.s;
        (((batch * self.h + head) * self.s.div_ceil(64) + t / 64) * 64 + t % 64) * 64
    }
    fn point(&self, row: usize, head: usize) -> Vec<f64> {
        let d = self.h * 64;
        let mut v = Vec::with_capacity(257);
        for plane in 0..4 {
            for dim in 0..64 {
                v.push(f64::from(
                    self.qkv[row * self.stride + plane * d + head * 64 + dim],
                ));
            }
        }
        v.push(f64::from(self.qkv[row * self.stride + 4 * d + head]));
        v
    }
    fn reference(&self, row: usize, head: usize, accumulate: bool, x: &[f64]) -> Vec<f64> {
        let at = self.compact(row, head);
        let aq = (0..64).map(|j| silu(x[j])).collect::<Vec<_>>();
        let ak = (0..64).map(|j| silu(x[64 + j])).collect::<Vec<_>>();
        let qn = (aq.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt();
        let kn = (ak.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt();
        let qd = qn + 1e-6;
        let kd = kn + 1e-6;
        let qdot = (0..64)
            .map(|j| aq[j] * f64::from(self.dq[at + j]))
            .sum::<f64>();
        let kdot = (0..64)
            .map(|j| ak[j] * f64::from(self.dk[at + j]))
            .sum::<f64>();
        let beta = sigmoid(x[256]);
        let mut result = vec![0.0; 257];
        let mut db = f64::from(self.db[at / 64]);
        for j in 0..64 {
            result[j] = 0.125
                * (f64::from(self.dq[at + j]) / qd - aq[j] * qdot / (qn * qd * qd))
                * silu_grad(x[j]);
            result[64 + j] = (f64::from(self.dk[at + j]) / kd - ak[j] * kdot / (kn * kd * kd))
                * silu_grad(x[64 + j]);
            result[128 + j] = beta * f64::from(self.dv[at + j]) * silu_grad(x[128 + j])
                + if accumulate {
                    f64::from(self.initial[row * self.stride + 2 * self.h * 64 + head * 64 + j])
                } else {
                    0.0
                };
            result[192 + j] = -0.01 * f64::from(self.dg[at + j]) * sigmoid(x[192 + j]);
            db += silu(x[128 + j]) * f64::from(self.dv[at + j]);
        }
        result[256] = db * beta * (1.0 - beta);
        result
    }
    fn objective(&self, row: usize, head: usize, accumulate: bool, x: &[f64]) -> f64 {
        let at = self.compact(row, head);
        let aq = (0..64).map(|j| silu(x[j])).collect::<Vec<_>>();
        let ak = (0..64).map(|j| silu(x[64 + j])).collect::<Vec<_>>();
        let qn = (aq.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt() + 1e-6;
        let kn = (ak.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt() + 1e-6;
        let beta = sigmoid(x[256]);
        let mut result = f64::from(self.db[at / 64]) * beta;
        for j in 0..64 {
            result += f64::from(self.dq[at + j]) * 0.125 * aq[j] / qn
                + f64::from(self.dk[at + j]) * ak[j] / kn
                - 0.01 * f64::from(self.dg[at + j]) * softplus(x[192 + j])
                + f64::from(self.dv[at + j]) * silu(x[128 + j]) * beta;
            if accumulate {
                result +=
                    f64::from(self.initial[row * self.stride + 2 * self.h * 64 + head * 64 + j])
                        * x[128 + j];
            }
        }
        result
    }
}
struct Inputs {
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
}
impl Inputs {
    fn new(s: &CudaStream, f: &Fixture) -> Result<Self, cuda_core::DriverError> {
        Ok(Self {
            qkv: DeviceBuffer::from_host(s, &f.qkv)?,
            q: DeviceBuffer::from_host(s, &f.dq)?,
            k: DeviceBuffer::from_host(s, &f.dk)?,
            g: DeviceBuffer::from_host(s, &f.dg)?,
            v: DeviceBuffer::from_host(s, &f.dv)?,
            b: DeviceBuffer::from_host(s, &f.db)?,
        })
    }
}
struct Outputs {
    qkv: DeviceBuffer<f32>,
    amax: DeviceBuffer<f32>,
}
impl Outputs {
    fn new(s: &CudaStream, f: &Fixture) -> Result<Self, cuda_core::DriverError> {
        let mut qkv = f.initial.clone();
        qkv.extend(std::iter::repeat_n(f32::from_bits(0x7fc01234), GUARD));
        let amax =
            vec![f32::from_bits(0x3eaa1234); f.b * f.h * f.s.div_ceil(64) * 8 + f.b * f.s + GUARD];
        Ok(Self {
            qkv: DeviceBuffer::from_host(s, &qkv)?,
            amax: DeviceBuffer::from_host(s, &amax)?,
        })
    }
    fn run(
        &mut self,
        m: &KdaExactRawVjpModule,
        s: &CudaStream,
        f: &Fixture,
        i: &Inputs,
        accumulate: bool,
    ) -> Result<u32, cuda_core::DriverError> {
        m.vjp(RawVjpArgs {
            stream: s,
            qkv: &i.qkv,
            dq: &i.q,
            dk: &i.k,
            dg: &i.g,
            dvbeta: &i.v,
            dbeta: &i.b,
            d_qkv: &mut self.qkv,
            d_qkv_chunk_amax: &mut self.amax,
            accumulate_value_grad: accumulate,
            batch_size: f.b as u32,
            seq_len: f.s as u32,
            head_count: f.h as u32,
            qkv_dim: f.stride as u32,
        })
    }
}
fn verify(f: &Fixture, actual: &[f32], amax: &[f32], accumulate: bool) -> (f64, usize) {
    let d = f.h * 64;
    let chunks = f.s.div_ceil(64);
    let mut max_error = 0.0f64;
    let mut tiny_nonzero = 0usize;
    for row in 0..f.b * f.s {
        for head in 0..f.h {
            let x = f.point(row, head);
            let expected = f.reference(row, head, accumulate, &x);
            let case = f.forced.unwrap_or((row / f.s * f.h + head) % 8);
            for (i, &target) in expected.iter().enumerate() {
                let index = if i < 256 {
                    row * f.stride + (i / 64) * d + head * 64 + i % 64
                } else {
                    row * f.stride + 4 * d + head
                };
                let value = f64::from(actual[index]);
                let error = (value - target).abs();
                max_error = max_error.max(error);
                let tolerance = if case % 2 == 0 { 2e-12 } else { 2e-6 };
                assert!(
                    value.is_finite() && error <= tolerance + 5e-4 * target.abs(),
                    "raw original FP32 VJP accumulate={accumulate} row={row} head={head} plane={} dim={} actual={value} expected={target} error={error}",
                    i / 64,
                    i % 64
                );
                if !accumulate && case % 2 == 0 && target.abs() > 1e-12 {
                    assert_ne!(value, 0.0, "discarded tiny raw cotangent");
                    tiny_nonzero += 1;
                }
            }
        }
        for column in 4 * d + f.h..f.stride {
            assert_eq!(
                actual[row * f.stride + column].to_bits(),
                f.initial[row * f.stride + column].to_bits(),
                "raw finish changed output-gate/padding field"
            );
        }
    }
    for bh in 0..f.b * f.h {
        let batch = bh / f.h;
        let head = bh % f.h;
        for chunk in 0..chunks {
            let mut maximum = 0.0f32;
            for t in chunk * 64..((chunk + 1) * 64).min(f.s) {
                let row = batch * f.s + t;
                for plane in 0..4 {
                    for dim in 0..64 {
                        maximum =
                            maximum.max(actual[row * f.stride + plane * d + head * 64 + dim].abs());
                    }
                }
                maximum = maximum.max(actual[row * f.stride + 4 * d + head].abs());
            }
            let start = (bh * chunks + chunk) * 8;
            assert_eq!(
                amax[start].to_bits(),
                maximum.to_bits(),
                "actual completed raw amax"
            );
            assert!(amax[start + 1..start + 8].iter().all(|x| x.to_bits() == 0));
        }
    }
    assert!(
        actual[f.b * f.s * f.stride..]
            .iter()
            .all(|x| x.to_bits() == 0x7fc01234)
    );
    assert!(
        amax[f.b * f.h * chunks * 8..]
            .iter()
            .all(|x| x.to_bits() == 0x3eaa1234)
    );
    (max_error, tiny_nonzero)
}

#[test]
#[ignore = "native B8 original raw Q/K/V/G/beta VJP with XSA value accumulation and actual amax"]
fn native_kda_exact_raw_vjp_original_precision_value_accumulation_and_amax()
-> Result<(), Box<dyn Error>> {
    let (_, stream, module) = common::cuda_test_module(KdaExactRawVjpModule::from_module)?;
    let f = Fixture::new(8, 2048, 32, 8320, None);
    let input = Inputs::new(&stream, &f)?;
    let mut max_error = 0.0f64;
    let mut tiny_nonzero = 0usize;
    for accumulate in [false, true] {
        let mut output = Outputs::new(&stream, &f)?;
        let count = output.run(&module, &stream, &f, &input, accumulate)?;
        assert_eq!(count, 8192 * 8);
        let actual = output.qkv.to_host_vec(&stream)?;
        let amax = output.amax.to_host_vec(&stream)?;
        let (error, tiny) = verify(&f, &actual, &amax, accumulate);
        max_error = max_error.max(error);
        tiny_nonzero += tiny;
        // An original8.0039 beta projects to half8.0. Require the actual raw
        // result to follow the original FP32 input rather than the saved tape.
        let x = f.point(0, 5);
        let mut rounded = x.clone();
        rounded[256] = half(x[256]);
        let exact = f.reference(0, 5, accumulate, &x)[256];
        let wrong = f.reference(0, 5, accumulate, &rounded)[256];
        let got = f64::from(actual[4 * 2048 + 5]);
        assert!((exact - wrong).abs() > 2e-5);
        assert!(
            (got - exact).abs() < (got - wrong).abs() / 4.0,
            "raw VJP substituted rounded beta tape"
        );
    }
    assert!(tiny_nonzero > 1_000_000);
    for len in [1usize, 15, 16, 17, 63, 64, 65, 127] {
        let f = Fixture::new(2, len, 2, 528, None);
        let input = Inputs::new(&stream, &f)?;
        let mut output = Outputs::new(&stream, &f)?;
        let count = output.run(&module, &stream, &f, &input, true)?;
        assert_eq!(count, 4 * len.div_ceil(64) as u32 * 8);
        verify(
            &f,
            &output.qkv.to_host_vec(&stream)?,
            &output.amax.to_host_vec(&stream)?,
            true,
        );
    }
    let mut max_fd_error = 0.0f64;
    let mut max_native_fd_error = 0.0f64;
    let coords = [
        0usize, 17, 63, 65, 95, 127, 128, 145, 191, 192, 225, 255, 256,
    ];
    for case in [0usize, 1, 2, 5] {
        let f = Fixture::new(1, 1, 1, 288, Some(case));
        let input = Inputs::new(&stream, &f)?;
        let mut output = Outputs::new(&stream, &f)?;
        output.run(&module, &stream, &f, &input, true)?;
        let actual = output.qkv.to_host_vec(&stream)?;
        let x = f.point(0, 0);
        let expected = f.reference(0, 0, true, &x);
        for &i in &coords {
            for step in [1e-6, 5e-7] {
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[i] += step;
                minus[i] -= step;
                let fd = (f.objective(0, 0, true, &plus) - f.objective(0, 0, true, &minus))
                    / (2.0 * step);
                let native = f64::from(actual[i]);
                max_fd_error = max_fd_error.max((fd - expected[i]).abs());
                max_native_fd_error = max_native_fd_error.max((fd - native).abs());
                assert!(
                    (fd - expected[i]).abs() <= 2e-7 + 5e-5 * expected[i].abs(),
                    "raw independent FD case={case} i={i} fd={fd} expected={}",
                    expected[i]
                );
                assert!(
                    (fd - native).abs() <= 3e-6 + 1e-3 * fd.abs(),
                    "raw native FD case={case} i={i} fd={fd} native={native}"
                );
            }
        }
    }
    let small = Fixture::new(1, 1, 1, 288, None);
    let si = Inputs::new(&stream, &small)?;
    let mut out = Outputs::new(&stream, &small)?;
    let before = out.qkv.to_host_vec(&stream)?;
    let before_amax = out.amax.to_host_vec(&stream)?;
    assert!(out.run(&module, &stream, &f, &si, false).is_err());
    for (b, s, h, stride) in [
        (0usize, 1, 1, 288),
        (1, 0, 1, 288),
        (1, 1, 0, 288),
        (1, 1, 1, 256),
        (u32::MAX as usize, 2048, 32, 8320),
        (1, u32::MAX as usize, 1, 288),
    ] {
        let mut bad = small.clone();
        bad.b = b;
        bad.s = s;
        bad.h = h;
        bad.stride = stride;
        assert!(out.run(&module, &stream, &bad, &si, false).is_err());
    }
    assert!(
        out.qkv
            .to_host_vec(&stream)?
            .iter()
            .zip(&before)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        out.amax
            .to_host_vec(&stream)?
            .iter()
            .zip(&before_amax)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!(
        "native_exact_KDA_raw_VJP_verified B=8 S=2048 H=32 chunk_batch=8192 original_FP32_affine=true value_accumulation_modes=2 partial_lengths=8 output_gate_fields=unchanged amax=actual_completed_gradient tiny_nonzero={tiny_nonzero} STE_coordinates=52 STE_step_sizes=2 max_error={max_error:e} max_analytic_FD_error={max_fd_error:e} max_native_FD_error={max_native_fd_error:e} guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
