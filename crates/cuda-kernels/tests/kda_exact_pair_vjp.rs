//! Declared before implementation: all8192chunks/eight patterns, actual rounded
//! off-diagonal branches and both log intervals, beta0/1, sub-FP16 upstreams,
//! independent direct-support oracle, two-step rounding-aware finite differences,
//! exact gradient-prefix causality, literal-zero future support, poisoned
//! padding, output guards and invalid host shapes. Host code never trains.
use cuda_core::{CudaStream, DeviceBuffer};
use rust_kernels_cuda::kda_exact_pair_vjp::{KdaExactPairVjpModule, PairVjpArgs};
use std::error::Error;
mod common;
const D: usize = 64;
const N: usize = 4096;
const CHUNKS: usize = 8192;
const GUARD: usize = 257;

#[derive(Clone)]
struct Fixture {
    q: Vec<f64>,
    k: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    dl: Vec<f64>,
    dc: Vec<f64>,
}
struct Grad {
    q: Vec<f64>,
    k: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
}
impl Fixture {
    fn new(case: usize, features: usize, tiny: bool) -> Self {
        let mut f = Self {
            q: vec![0.0; N],
            k: vec![0.0; N],
            g: vec![0.0; N],
            b: vec![0.0; D],
            dl: vec![f64::NAN; N],
            dc: vec![f64::NAN; N],
        };
        let scale = 2.0f64.powi(if tiny { -28 } else { -8 });
        for t in 0..D {
            f.b[t] = match case {
                6 => 0.0,
                7 => 1.0,
                _ => (1 + t % 4) as f64 / 8.0,
            };
            for h in 0..features {
                let i = t * D + h;
                f.q[i] = if case == 2 {
                    1.0 / 64.0
                } else {
                    ((t * 13 + h * 7 + case) as i32 % 9 - 4) as f64 / 256.0
                };
                f.k[i] = if case == 2 {
                    1.0 / 8.0
                } else {
                    ((t * 5 + h * 3 + case) as i32 % 5 - 2) as f64 / 16.0
                };
                f.g[i] = f64::from(match case {
                    0 => -0.01f32,
                    1 => -0.3,
                    2 => {
                        if t == 0 {
                            -600.0
                        } else {
                            -0.0001
                        }
                    }
                    3 => -8.0,
                    4 => 0.0,
                    _ => -0.015 * (1 + (t + h * 3) % 7) as f32,
                });
            }
            for j in 0..=t {
                if j < t {
                    f.dl[t * D + j] = (((t * 3 + j * 11 + case) % 9) as i32 - 4) as f64 * scale;
                }
                f.dc[t * D + j] = (((t * 13 + j * 7 + case) % 11) as i32 - 5) as f64 * scale;
            }
        }
        f
    }
}
fn half(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    let spacing = 2.0f64.powi((x.abs().log2().floor() as i32).max(-14) - 10);
    (x / spacing).round_ties_even() * spacing
}
// Match the FP32 interval producer's accumulation order, while differentiating
// original increments under the declared ordinary FP32/FP16 cast STE.
fn log(base: &Fixture, first: usize, last: usize, h: usize, reverse: bool) -> f64 {
    let mut sum = 0.0f32;
    if reverse {
        for s in (first..=last).rev() {
            sum += base.g[s * D + h] as f32;
        }
    } else {
        for s in first..=last {
            sum += base.g[s * D + h] as f32;
        }
    }
    f64::from(sum)
}
fn reference(f: &Fixture, valid: usize, features: usize) -> Grad {
    let mut d = Grad {
        q: vec![0.0; N],
        k: vec![0.0; N],
        g: vec![0.0; N],
        b: vec![0.0; D],
    };
    for t in 0..valid {
        for j in 0..=t {
            let dl = if j < t { f.dl[t * D + j] } else { 0.0 };
            let dc = f.dc[t * D + j];
            if dc == 0.0 && dl == 0.0 {
                continue;
            }
            for h in 0..features {
                let ti = t * D + h;
                let ji = j * D + h;
                let qt = f.q[ti];
                let kt = f.k[ti];
                let kj = f.k[ji];
                let b = f.b[t];
                if t / 16 == j / 16 {
                    let e = log(f, j + 1, t, h, true).exp();
                    d.q[ti] += dc * kj * e;
                    d.k[ti] += dl * b * kj * e;
                    d.k[ji] += (dc * qt + dl * b * kt) * e;
                    d.b[t] += dl * kt * kj * e;
                    let dg = (dc * qt * kj + dl * b * kt * kj) * e;
                    // Independent explicit support accumulation, not difference events.
                    for s in j + 1..=t {
                        d.g[s * D + h] += dg;
                    }
                } else {
                    let anchor = t / 16 * 16;
                    let eq = log(f, anchor + 1, t, h, false).exp();
                    let ek = log(f, j + 1, anchor, h, false).exp();
                    let aq = half(qt * eq);
                    let al = half(b * kt * eq);
                    let bk = half(kj * ek);
                    let daq = dc * bk;
                    let dal = dl * bk;
                    let dbk = dc * aq + dl * al;
                    d.q[ti] += eq * daq;
                    d.k[ti] += b * eq * dal;
                    d.k[ji] += ek * dbk;
                    d.b[t] += kt * eq * dal;
                    let gq = qt * eq * daq + b * kt * eq * dal;
                    let gk = kj * ek * dbk;
                    for s in anchor + 1..=t {
                        d.g[s * D + h] += gq;
                    }
                    for s in j + 1..=anchor {
                        d.g[s * D + h] += gk;
                    }
                }
            }
        }
    }
    d
}
// Smooth STE witness. Freeze rounding residuals at the original point, including
// the FP32 interval sums and FP16 factors. Fresh quantization at each perturbed
// point would test a staircase rather than the selected straight-through map.
fn ste_objective(changed: &Fixture, base: &Fixture, valid: usize, features: usize) -> f64 {
    let interval = |first: usize, last: usize, h: usize, reverse: bool| {
        let delta = (first..=last)
            .map(|s| changed.g[s * D + h] - base.g[s * D + h])
            .sum::<f64>();
        (log(base, first, last, h, reverse) + delta).exp()
    };
    let mut objective = 0.0;
    for t in 0..valid {
        for j in 0..=t {
            let dl = if j < t { base.dl[t * D + j] } else { 0.0 };
            let dc = base.dc[t * D + j];
            for h in 0..features {
                let ti = t * D + h;
                let ji = j * D + h;
                if t / 16 == j / 16 {
                    let e = interval(j + 1, t, h, true);
                    objective += (dc * changed.q[ti] * changed.k[ji]
                        + dl * changed.b[t] * changed.k[ti] * changed.k[ji])
                        * e;
                } else {
                    let a = t / 16 * 16;
                    let eq0 = log(base, a + 1, t, h, false).exp();
                    let ek0 = log(base, j + 1, a, h, false).exp();
                    let eq = interval(a + 1, t, h, false);
                    let ek = interval(j + 1, a, h, false);
                    let aq = changed.q[ti] * eq + (half(base.q[ti] * eq0) - base.q[ti] * eq0);
                    let al = changed.b[t] * changed.k[ti] * eq
                        + (half(base.b[t] * base.k[ti] * eq0) - base.b[t] * base.k[ti] * eq0);
                    let bk = changed.k[ji] * ek + (half(base.k[ji] * ek0) - base.k[ji] * ek0);
                    objective += (dc * aq + dl * al) * bk;
                }
            }
        }
    }
    objective
}

struct DeviceInputs {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    dl: DeviceBuffer<f32>,
    dc: DeviceBuffer<f32>,
}
struct DeviceGrads {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
}
impl DeviceInputs {
    fn from_fixture(stream: &CudaStream, f: &Fixture) -> Result<Self, cuda_core::DriverError> {
        let to_dev = |v: &Vec<f64>| {
            DeviceBuffer::from_host(stream, &v.iter().map(|&x| x as f32).collect::<Vec<_>>())
        };
        Ok(Self {
            q: to_dev(&f.q)?,
            k: to_dev(&f.k)?,
            g: to_dev(&f.g)?,
            b: to_dev(&f.b)?,
            dl: to_dev(&f.dl)?,
            dc: to_dev(&f.dc)?,
        })
    }
}
impl DeviceGrads {
    fn new(stream: &CudaStream, n: usize) -> Result<Self, cuda_core::DriverError> {
        let buffer =
            |n| DeviceBuffer::from_host(stream, &vec![f32::from_bits(0x7fc01234); n + GUARD]);
        Ok(Self {
            q: buffer(n)?,
            k: buffer(n)?,
            g: buffer(n)?,
            b: buffer(n / 64)?,
        })
    }
    fn run(
        &mut self,
        module: &KdaExactPairVjpModule,
        stream: &CudaStream,
        f: &DeviceInputs,
        bh: u32,
        seq_len: u32,
    ) -> Result<(), cuda_core::DriverError> {
        module.vjp(PairVjpArgs {
            stream,
            q: &f.q,
            k: &f.k,
            g: &f.g,
            beta: &f.b,
            lower_grad: &f.dl,
            readout_grad: &f.dc,
            dq: &mut self.q,
            dk: &mut self.k,
            dg: &mut self.g,
            dbeta: &mut self.b,
            batch_heads: bh,
            seq_len,
        })
    }
    fn host(&self, stream: &CudaStream) -> Result<[Vec<f32>; 4], cuda_core::DriverError> {
        Ok([
            self.q.to_host_vec(stream)?,
            self.k.to_host_vec(stream)?,
            self.g.to_host_vec(stream)?,
            self.b.to_host_vec(stream)?,
        ])
    }
}

#[test]
#[ignore = "complete native8192 rounded pair VJP, finite differences and causality"]
fn native_kda_exact_pair_vjp_all_branches_tiny_gradients_and_prefixes() -> Result<(), Box<dyn Error>>
{
    let (_, stream, module) = common::cuda_test_module(KdaExactPairVjpModule::from_module)?;
    let cases = (0..8)
        .map(|case| Fixture::new(case, 64, case % 2 == 0))
        .collect::<Vec<_>>();
    let refs = cases
        .iter()
        .map(|f| reference(f, 64, 64))
        .collect::<Vec<_>>();
    let mut all = Fixture {
        q: Vec::with_capacity(CHUNKS * N),
        k: Vec::with_capacity(CHUNKS * N),
        g: Vec::with_capacity(CHUNKS * N),
        b: Vec::with_capacity(CHUNKS * D),
        dl: Vec::with_capacity(CHUNKS * N),
        dc: Vec::with_capacity(CHUNKS * N),
    };
    for chunk in 0..CHUNKS {
        let f = &cases[chunk % 8];
        let scale = (chunk + 1) as f64 / 8192.0;
        all.q.extend(&f.q);
        all.k.extend(&f.k);
        all.g.extend(&f.g);
        all.b.extend(&f.b);
        all.dl.extend(f.dl.iter().map(|&x| x * scale));
        all.dc.extend(f.dc.iter().map(|&x| x * scale));
    }
    let inputs = DeviceInputs::from_fixture(&stream, &all)?;
    drop(all);
    let mut out = DeviceGrads::new(&stream, CHUNKS * N)?;
    out.run(&module, &stream, &inputs, 256, 2048)?;
    let full = out.host(&stream)?;
    let mut max_errors = [0.0f64; 4];
    let mut tiny_nonzero = 0usize;
    for chunk in 0..CHUNKS {
        let scale = (chunk + 1) as f64 / 8192.0;
        let r = &refs[chunk % 8];
        for (plane, expected) in [&r.q, &r.k, &r.g, &r.b].iter().enumerate() {
            let count = if plane == 3 { D } else { N };
            for i in 0..count {
                let value = f64::from(full[plane][chunk * count + i]);
                let target = expected[i] * scale;
                let error = (value - target).abs();
                max_errors[plane] = max_errors[plane].max(error);
                let tolerance = if chunk % 2 == 0 { 2e-12 } else { 2e-6 };
                assert!(
                    value.is_finite() && error <= tolerance + 2e-4 * target.abs(),
                    "pair VJP chunk={chunk} plane={plane} i={i} actual={value} expected={target} error={error}"
                );
                if chunk % 2 == 0 && target.abs() > 1e-12 {
                    assert_ne!(value, 0.0, "discarded tiny pair cotangent");
                    tiny_nonzero += 1;
                }
            }
        }
        assert!(
            full[2][chunk * N..chunk * N + D]
                .iter()
                .all(|x| x.to_bits() == 0),
            "pair factors cannot depend on g0"
        );
    }
    assert!(tiny_nonzero > 1_000_000);
    for (plane, values) in full.iter().enumerate() {
        let count = if plane == 3 { CHUNKS * D } else { CHUNKS * N };
        assert!(values[count..].iter().all(|x| x.to_bits() == 0x7fc01234));
    }
    // Restricted-prefix losses: both full and truncated shapes must give exact
    // past bits and literal0 for all future q/k/g/beta coordinates.
    for cut in [1usize, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64] {
        let mut f = Fixture::new(5, 64, false);
        for t in cut..64 {
            for j in 0..=t {
                if j < t {
                    f.dl[t * D + j] = 0.0;
                }
                f.dc[t * D + j] = 0.0;
            }
        }
        let original = DeviceInputs::from_fixture(&stream, &f)?;
        let mut small = DeviceGrads::new(&stream, N)?;
        small.run(&module, &stream, &original, 1, 64)?;
        let before = small.host(&stream)?;
        for t in cut..64 {
            f.q[t * D..(t + 1) * D].fill(0.375);
            f.k[t * D..(t + 1) * D].fill(-0.25);
            f.g[t * D..(t + 1) * D].fill(-31.0);
            f.b[t] = 0.25;
        }
        let changed = DeviceInputs::from_fixture(&stream, &f)?;
        small.run(&module, &stream, &changed, 1, 64)?;
        let after = small.host(&stream)?;
        for plane in 0..4 {
            let end = if plane == 3 { cut } else { cut * D };
            let total = if plane == 3 { D } else { N };
            assert!(
                before[plane][..end]
                    .iter()
                    .zip(&after[plane][..end])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "future changed pair gradient prefix cut={cut} plane={plane}"
            );
            assert!(
                after[plane][end..total].iter().all(|x| x.to_bits() == 0),
                "future gradient support cut={cut} plane={plane}"
            );
        }
        for t in cut..64 {
            f.q[t * D..(t + 1) * D].fill(f64::NAN);
            f.k[t * D..(t + 1) * D].fill(f64::NAN);
            f.g[t * D..(t + 1) * D].fill(f64::NAN);
            f.b[t] = f64::NAN;
        }
        let padded = DeviceInputs::from_fixture(&stream, &f)?;
        small.run(&module, &stream, &padded, 1, cut as u32)?;
        let got = small.host(&stream)?;
        for plane in 0..4 {
            let total = if plane == 3 { D } else { N };
            assert!(
                got[plane][..total]
                    .iter()
                    .zip(&before[plane][..total])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "truncated/poisoned gradient cut={cut} plane={plane}"
            );
        }
    }
    // Native values against the separately evaluated smooth STE finite
    // difference, including diagonal, three anchors and both log branches.
    let coords = [
        (0, 0, 1),
        (0, 16, 2),
        (0, 32, 3),
        (0, 48, 0),
        (1, 0, 2),
        (1, 15, 1),
        (1, 31, 3),
        (1, 47, 0),
        (2, 0, 0),
        (2, 1, 1),
        (2, 16, 2),
        (2, 17, 3),
        (2, 32, 1),
        (2, 48, 2),
        (3, 16, 0),
        (3, 48, 0),
    ];
    let mut max_fd_error = 0.0f64;
    let mut max_native_fd_error = 0.0f64;
    for case in [0usize, 1, 2, 5] {
        let f = Fixture::new(case, 4, false);
        let analytic = reference(&f, 49, 4);
        let device = DeviceInputs::from_fixture(&stream, &f)?;
        let mut grads = DeviceGrads::new(&stream, N)?;
        grads.run(&module, &stream, &device, 1, 49)?;
        let native = grads.host(&stream)?;
        for &(plane, t, h) in &coords {
            let i = if plane == 3 { t } else { t * D + h };
            let target = [&analytic.q, &analytic.k, &analytic.g, &analytic.b][plane][i];
            for step in [1e-4, 5e-5] {
                let mut plus = f.clone();
                let mut minus = f.clone();
                let a = match plane {
                    0 => &mut plus.q,
                    1 => &mut plus.k,
                    2 => &mut plus.g,
                    _ => &mut plus.b,
                };
                a[i] += step;
                let b = match plane {
                    0 => &mut minus.q,
                    1 => &mut minus.k,
                    2 => &mut minus.g,
                    _ => &mut minus.b,
                };
                b[i] -= step;
                let fd = (ste_objective(&plus, &f, 49, 4) - ste_objective(&minus, &f, 49, 4))
                    / (2.0 * step);
                max_fd_error = max_fd_error.max((fd - target).abs());
                max_native_fd_error =
                    max_native_fd_error.max((fd - f64::from(native[plane][i])).abs());
                assert!(
                    (fd - target).abs() <= 2e-8 + 1e-5 * target.abs(),
                    "independent STE finite difference case={case} plane={plane} t={t} h={h} fd={fd} reference={target}"
                );
                assert!(
                    (fd - f64::from(native[plane][i])).abs() <= 2e-6 + 5e-4 * fd.abs(),
                    "native STE finite difference case={case} plane={plane} t={t} h={h}"
                );
            }
        }
    }
    for (bh, seq_len) in [(0, 2048), (256, 0), (u32::MAX, 2048), (256, u32::MAX)] {
        assert!(out.run(&module, &stream, &inputs, bh, seq_len).is_err());
    }
    let short = DeviceInputs::from_fixture(&stream, &cases[0])?;
    assert!(out.run(&module, &stream, &short, 256, 2048).is_err());
    let unchanged = out.host(&stream)?;
    for plane in 0..4 {
        assert!(
            unchanged[plane]
                .iter()
                .zip(&full[plane])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    println!(
        "native_exact_KDA_pair_VJP_verified chunk_batch={CHUNKS} pattern_cases=8 tiny_nonzero={tiny_nonzero} prefix_cuts=12 literal_zero_future=true STE_coordinates=64 STE_step_sizes=2 max_errors={max_errors:?} max_analytic_FD_error={max_fd_error:e} max_native_FD_error={max_native_fd_error:e} guards=unchanged optimizer_updates=0"
    );
    Ok(())
}
